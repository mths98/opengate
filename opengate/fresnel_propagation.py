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

# h * c in keV * mm
HC_KEV_MM = 12.398419843320026e-7

MODELS = ("article", "physical")


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
    e_eff = meta.get("effective_energy_keV")
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

    "article": Psi = Re + i Im, as recorded.
    "physical": Psi_f = sqrt(sum E) Psi / |Psi|, with 0 where |Psi| = 0.
    """
    psi = data.real + 1j * data.imag
    if model == "article":
        return psi
    if model == "physical":
        mod = np.abs(psi)
        out = np.zeros_like(psi)
        np.divide(psi, mod, out=out, where=mod > 0)
        out *= np.sqrt(np.clip(data.energy, 0.0, None))
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
        json.dump(info, f, indent=2)
