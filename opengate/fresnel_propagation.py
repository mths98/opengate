"""Free-space Fresnel propagation of the wavefront recorded by the WavefrontActor.

Post-processing of the complex field on the detector plane (Eq. 11, 14 and 15 of
Sanchez-Lara et al. 2026). This module holds the core utilities: reading the
files written by the WavefrontActor, the two amplitude models, the wavelength,
the sampling check, the cone-beam scaling, the intensity, the block reduction
and the writing of the result.

Axis convention: arrays have shape (ny, nx) (row = y); every pair of values
(spacing, corner, origin) is given in the order (y, x), matching the array axes.
All lengths are in mm and energies in keV.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import itk
import numpy as np
import scipy.fft

# h * c in keV * mm
HC_KEV_MM = 12.398419843320026e-7

MODELS = ("article", "physical")

PADDINGS = {"none": None, "zero": "constant", "mirror": "symmetric"}

# keys of the JSON file of the WavefrontActor that the propagation needs
REQUIRED_KEYS = (
    "plane_z_mm",
    "n_photons_counted",
    "spacing_mm",
    "grid_corner_mm",
    "effective_energy_keV",
)


@dataclass
class WavefrontData:
    """Wavefront read from disk. Pairs are in (y, x) order."""

    real: np.ndarray
    imag: np.ndarray
    energy: np.ndarray
    spacing_mm: tuple
    grid_corner_mm: tuple
    plane_z_mm: float
    effective_energy_keV: float
    n_photons_counted: int
    meta: dict = field(default_factory=dict)


def _base(base_path):
    """Strip a trailing .mhd or .json, so that both '<base>' and '<base>.mhd'
    are accepted."""
    p = Path(base_path)
    if p.suffix in (".mhd", ".json"):
        p = p.with_suffix("")
    return p


def read_wavefront(base_path):
    """Read <base>_real.mhd, <base>_imag.mhd, <base>_energy.mhd and <base>.json.

    base_path is either the prefix <base> or the base file name <base>.mhd (the
    output_filename of the actor); a trailing .mhd or .json is removed.
    """
    base = _base(base_path)
    files = {
        s: Path(f"{base}_{s}.mhd") for s in ("real", "imag", "energy")
    }
    files["json"] = Path(f"{base}.json")
    for name, p in files.items():
        if not p.is_file():
            raise FileNotFoundError(
                f"Incomplete wavefront: file '{p}' ({name}) not found."
            )
    with open(files["json"]) as f:
        meta = json.load(f)
    missing = [k for k in REQUIRED_KEYS if k not in meta]
    if missing:
        raise ValueError(
            f"'{files['json']}': missing required key(s) {', '.join(missing)}."
        )
    e_eff = meta["effective_energy_keV"]
    if e_eff is None:
        raise ValueError(
            f"'{files['json']}': effective_energy_keV is null (no photon was "
            "counted, N = 0); there is no wavefront to propagate."
        )
    grids = {}
    for s in ("real", "imag", "energy"):
        grids[s] = np.array(itk.array_from_image(itk.imread(str(files[s]))), "f8")
        if grids[s].ndim != 2:
            raise ValueError(f"'{files[s]}' is not a 2D image.")
    shape = grids["real"].shape
    if any(g.shape != shape for g in grids.values()):
        raise ValueError(f"The real, imag and energy grids of '{base}' differ.")
    dx, dy = meta["spacing_mm"]
    x0, y0 = meta["grid_corner_mm"]
    return WavefrontData(
        real=grids["real"],
        imag=grids["imag"],
        energy=grids["energy"],
        spacing_mm=(float(dy), float(dx)),
        grid_corner_mm=(float(y0), float(x0)),
        plane_z_mm=float(meta["plane_z_mm"]),
        effective_energy_keV=float(e_eff),
        n_photons_counted=int(meta["n_photons_counted"]),
        meta=meta,
    )


def build_field(data, model):
    """Complex field on the plane.

    "article": Psi = Re + i Im, as recorded (Eq. 13): for N photons of energy E
    with the mean phasor <e^{i Phi}> in the pixel, |Psi|^2 = N^2 E |<e^{i Phi}>|^2,
    so the intensity grows as N^2 (the attenuation is doubled, D24).
    "physical": Psi_f = Psi / sqrt(N), with N = sum E / E_eff the estimated
    number of photons of the pixel (E_eff = effective_energy_keV), and 0 where
    sum E = 0. Then |Psi_f|^2 = N E |<e^{i Phi}>|^2: the energy N E (Beer-Lambert)
    times the loss of coherence inside the pixel, |<e^{i Phi}>| <= 1, which is
    the pixel average of the transmission function where the phase varies
    across the pixel (object edges). The former model sqrt(sum E) Psi / |Psi|
    kept only the phase of Psi and so discarded that loss of coherence; on a
    1 um grid it overestimated the edge fringes by 10-40 %.

    The deposited energy cannot be negative: values below -1e-12 max(E) raise a
    ValueError; smaller negative values are rounding errors and are set to 0.
    """
    psi = data.real + 1j * data.imag
    if model == "article":
        return psi
    if model == "physical":
        energy = data.energy
        tol = 1e-12 * max(float(np.max(energy)), 0.0)
        e_min = float(np.min(energy))
        if e_min < -tol:
            raise ValueError(
                f"The energy grid has negative values (min = {e_min:.6g}, "
                f"max = {float(np.max(energy)):.6g}); it is not a valid wavefront."
            )
        sqrt_n = np.sqrt(np.clip(energy, 0.0, None) / data.effective_energy_keV)
        out = np.zeros_like(psi)
        np.divide(psi, sqrt_n, out=out, where=sqrt_n > 0)
        return out
    raise ValueError(f"Unknown model '{model}', expected one of {MODELS}.")


def wavelength_mm(energy_keV):
    """lambda = h c / E, in mm."""
    if energy_keV <= 0:
        raise ValueError(f"The energy must be positive, got {energy_keV}.")
    return HC_KEV_MM / energy_keV


def fresnel_number(pixel_mm, wavelength_mm, distance_mm, n=10):
    """Fresnel number (n pixel)^2 / (lambda z) of a feature of n pixels."""
    return (n * pixel_mm) ** 2 / (wavelength_mm * distance_mm)


def check_sampling(pixel_mm, extent_mm, wavelength_mm, z_eff_mm):
    """Sampling criterion of the transfer-function propagator:
    pixel >= lambda z_eff / L, with L the extent of the grid (padding included).
    Only reports, never raises."""
    critical = wavelength_mm * z_eff_mm / extent_mm
    sqrt_lz = float(np.sqrt(wavelength_mm * z_eff_mm))
    ok = bool(pixel_mm >= critical)
    msg = (
        f"pixel = {pixel_mm:.4g} mm, critical pixel lambda z/L = {critical:.4g} mm, "
        f"sqrt(lambda z) = {sqrt_lz:.4g} mm: "
    )
    msg += (
        "sampling ok."
        if ok
        else "WARNING: under-sampled (pixel < lambda z/L); the result may alias."
    )
    return {
        "ok": ok,
        "critical_pixel_mm": float(critical),
        "sqrt_lambda_z_mm": sqrt_lz,
        "message": msg,
    }


def cone_beam_scaling(source_to_plane_mm, distance_mm):
    """(M, z_eff) with M = (R1 + D) / R1 and z_eff = D / M.
    source_to_plane_mm = None means a parallel beam: (1, D)."""
    if source_to_plane_mm is None:
        return 1.0, float(distance_mm)
    m = (source_to_plane_mm + distance_mm) / source_to_plane_mm
    return float(m), float(distance_mm / m)


def intensity(field, M=1.0):
    """|field|^2 / M^2 as float64."""
    return (field.real**2 + field.imag**2) / (M * M)


def downsample(image, factor):
    """Mean over blocks of factor x factor pixels; the factor must divide both
    dimensions."""
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"The reduction factor must be >= 1, got {factor}.")
    ny, nx = image.shape
    if ny % factor or nx % factor:
        raise ValueError(
            f"The reduction factor {factor} does not divide the image shape "
            f"({ny}, {nx})."
        )
    return image.reshape(ny // factor, factor, nx // factor, factor).mean(
        axis=(1, 3)
    )


def propagate(
    field,
    pixel_mm,
    wavelength_mm,
    distance_mm,
    padding="none",
    pad_factor=2,
    workers=-1,
):
    """Free-space Fresnel propagation by the transfer function (Eq. 14):

        Psi(z) = IFFT[ FFT[Psi(0)] H ],  H = exp(-i pi lambda z (fx^2 + fy^2)),

    without the global phase exp(i k z). The convention is the one of the recorded
    field: exp(-i w t), plane wave exp(+i k z), so a diverging wave has the phase
    +pi r^2 / (lambda R) and a converging lens -pi r^2 / (lambda f).

    field: 2D array of shape (ny, nx); pixel_mm = (dy, dx); the frequencies are
    scipy.fft.fftfreq(n, d) on each axis. For a cone beam, call it with the pixel
    of the plane and z_eff (see cone_beam_scaling).

    padding:
      - "none": the propagation is done in place in the complex128 array (the
        field itself if it is already a complex128 array), which is returned; the
        convolution is circular.
      - "zero" / "mirror": the grid is enlarged pad_factor times on each axis
        (np.pad, modes "constant" and "symmetric", centred), propagated and
        cropped back; a new array of the original shape is returned.

    distance_mm = 0 returns the field unchanged (a copy with padding).
    """
    if padding not in PADDINGS:
        raise ValueError(
            f"Unknown padding '{padding}', expected one of {tuple(PADDINGS)}."
        )
    if np.ndim(field) != 2:
        raise ValueError(f"The field must be a 2D array, got shape {np.shape(field)}.")
    dy, dx = (float(p) for p in pixel_mm)
    if dy <= 0 or dx <= 0:
        raise ValueError(f"The pixel must be positive, got (dy, dx) = {pixel_mm}.")
    if wavelength_mm <= 0:
        raise ValueError(f"The wavelength must be positive, got {wavelength_mm}.")
    pad_factor = int(pad_factor)
    if padding != "none" and pad_factor < 1:
        raise ValueError(f"pad_factor must be >= 1, got {pad_factor}.")

    if padding == "none":
        work = np.asarray(field, dtype=np.complex128)
    else:
        field = np.asarray(field, dtype=np.complex128)
        if distance_mm == 0:
            return field.copy()
        ny, nx = field.shape
        widths = []
        for n in (ny, nx):
            extra = (pad_factor - 1) * n
            widths.append((extra // 2, extra - extra // 2))
        work = np.pad(field, widths, mode=PADDINGS[padding])
    if distance_mm == 0:
        return work

    py, px = work.shape
    a = -np.pi * wavelength_mm * distance_mm
    hy = np.exp(1j * a * scipy.fft.fftfreq(py, dy) ** 2)
    hx = np.exp(1j * a * scipy.fft.fftfreq(px, dx) ** 2)
    work[...] = scipy.fft.fft2(work, workers=workers, overwrite_x=True)
    work *= hy[:, None]
    work *= hx[None, :]
    work[...] = scipy.fft.ifft2(work, workers=workers, overwrite_x=True)

    if padding == "none":
        return work
    (y0, _), (x0, _) = widths
    return work[y0 : y0 + ny, x0 : x0 + nx].copy()


def propagate_to_detector(
    field,
    pixel_mm,
    origin_mm,
    wavelength_mm,
    distance_mm,
    source_distance_mm=None,
    padding="none",
    pad_factor=2,
    workers=-1,
):
    """Intensity on the detector at distance_mm from the plane of the field.

    The single implementation of the recipe (cone beam: Fresnel scaling theorem):
    cone_beam_scaling -> propagate with z_eff and the pixel of the plane ->
    intensity / M^2 -> pixel and origin times M. source_distance_mm = None is a
    parallel beam (M = 1).

    pixel_mm and origin_mm are (y, x) pairs; origin_mm is the centre of the first
    pixel, relative to the optical axis x = y = 0 (which is the point that the
    magnification keeps fixed). With padding="none" the array field is modified
    in place (see propagate): pass a copy to keep it.

    Returns (image, detector_pixel_mm, detector_origin_mm, meta); meta holds
    distance_mm, z_eff_mm, M, wavelength_mm, fresnel_number_10px (for the pixel of
    the plane and z_eff) and the result of check_sampling (padding included).
    """
    m, z_eff = cone_beam_scaling(source_distance_mm, distance_mm)
    psi = propagate(
        field, pixel_mm, wavelength_mm, z_eff, padding, pad_factor, workers
    )
    image = intensity(psi, m)
    ny, nx = np.shape(field)
    factor = 1 if padding == "none" else int(pad_factor)
    # the criterion is checked on each axis; the worst one (pixel / critical) is kept
    checks = [
        check_sampling(pixel_mm[i], n * pixel_mm[i] * factor, wavelength_mm, z_eff)
        for i, n in enumerate((ny, nx))
    ]
    ratios = [c["critical_pixel_mm"] / pixel_mm[i] for i, c in enumerate(checks)]
    sampling = checks[int(np.argmax(ratios))]
    pix = max(pixel_mm)
    f10 = fresnel_number(pix, wavelength_mm, z_eff) if z_eff > 0 else None
    meta = {
        "distance_mm": float(distance_mm),
        "source_distance_mm": source_distance_mm,
        "z_eff_mm": z_eff,
        "M": m,
        "wavelength_mm": float(wavelength_mm),
        "fresnel_number_10px": f10,
        "padding": padding,
        "pad_factor": int(pad_factor) if padding != "none" else 1,
        "sampling": sampling,
    }
    det_pixel = (float(pixel_mm[0]) * m, float(pixel_mm[1]) * m)
    det_origin = (float(origin_mm[0]) * m, float(origin_mm[1]) * m)
    return image, det_pixel, det_origin, meta


def _json_default(obj):
    """Convert numpy scalars and arrays to native Python types for json.dump."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_intensity(path, image, pixel_mm, origin_mm, meta):
    """Write <path>.mhd (2D double) and <path>.json (with meta).

    pixel_mm = (dy, dx); origin_mm = (y, x) of the centre of the first pixel (the
    convention of the WavefrontActor). A trailing .mhd or .json in path is ignored.
    """
    base = _base(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    img = itk.image_view_from_array(np.ascontiguousarray(image, dtype=np.float64))
    img.SetSpacing([float(pixel_mm[1]), float(pixel_mm[0])])
    img.SetOrigin([float(origin_mm[1]), float(origin_mm[0])])
    itk.imwrite(img, f"{base}.mhd")
    info = {
        "mhd_offset_convention": "center of first pixel",
        "shape_yx": [int(image.shape[0]), int(image.shape[1])],
        "pixel_mm_yx": [float(pixel_mm[0]), float(pixel_mm[1])],
        "origin_mm_yx": [float(origin_mm[0]), float(origin_mm[1])],
        "meta": meta,
    }
    with open(f"{base}.json", "w") as f:
        json.dump(info, f, indent=2, default=_json_default)
