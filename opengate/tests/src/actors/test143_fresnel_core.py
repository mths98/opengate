#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests of the core of opengate.fresnel_propagation (no simulation).

A synthetic wavefront is written in the exact format of the WavefrontActor
(three 2D double MHD files with the origin at the centre of the first pixel,
and the JSON file), then read back and processed.
"""

import json
from pathlib import Path

import itk
import numpy as np

import opengate.tests.utility as utility
from opengate import fresnel_propagation as fp

HC_KEV_MM = 12.398419843320026e-7


def write_synthetic(base, ny=6, nx=4, spacing=(0.002, 0.001), energy=20.0, n=5):
    """Write a wavefront in the format of the WavefrontActor. spacing = (dy, dx)."""
    rng = np.random.default_rng(1)
    dy, dx = spacing
    corner = (-nx * dx / 2, -ny * dy / 2)  # (x0, y0) as in the JSON
    grids = {
        s: rng.normal(size=(ny, nx)) for s in ("real", "imag", "energy")
    }
    for s, g in grids.items():
        img = itk.image_view_from_array(g)
        img.SetSpacing([dx, dy])
        img.SetOrigin([corner[0] + dx / 2, corner[1] + dy / 2])
        itk.imwrite(img, f"{base}_{s}.mhd")
    meta = {
        "n_photons_counted": n,
        "n_outside_grid": 0,
        "n_recrossings": 0,
        "n_backward_crossings": 0,
        "effective_energy_keV": energy,
        "plane_z_mm": 10.0,
        "plane_center_mm": [0.0, 0.0],
        "grid_corner_mm": [corner[0], corner[1]],
        "mhd_offset_convention": "center of first pixel",
        "size": [nx, ny],
        "spacing_mm": [dx, dy],
        "kill_after_plane": False,
    }
    with open(f"{base}.json", "w") as f:
        json.dump(meta, f)
    return grids, meta


def raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:
        return str(e)
    return None


if __name__ == "__main__":
    paths = utility.get_default_test_paths(__file__, output_folder="test143_core")
    out = Path(paths.output)
    out.mkdir(parents=True, exist_ok=True)
    is_ok = True

    # (a) reading
    base = out / "synth"
    grids, meta = write_synthetic(str(base))
    for arg in (base, str(base), str(base) + ".mhd"):
        d = fp.read_wavefront(arg)
        ok = all(
            np.array_equal(getattr(d, s), grids[s]) for s in ("real", "imag", "energy")
        )
        ok &= d.real.shape == (6, 4)
        ok &= d.spacing_mm == (0.002, 0.001)
        ok &= d.grid_corner_mm == (meta["grid_corner_mm"][1], meta["grid_corner_mm"][0])
        ok &= d.effective_energy_keV == 20.0 and d.n_photons_counted == 5
        ok &= d.plane_z_mm == 10.0 and d.meta["size"] == [4, 6]
        is_ok &= utility.print_test(ok, f"(a) read_wavefront({arg!s:.40})")

    # (b) incomplete files, N = 0
    for missing in ("_real.mhd", "_imag.mhd", "_energy.mhd", ".json"):
        b = out / ("inc_" + missing.strip("._").replace(".", "_"))
        write_synthetic(str(b))
        Path(f"{b}{missing}").unlink()
        msg = raises(fp.read_wavefront, b)
        ok = msg is not None and missing in msg
        is_ok &= utility.print_test(ok, f"(b) missing {missing}: error = {msg[-60:]!r}")
    b = out / "n0"
    _, m = write_synthetic(str(b))
    m["effective_energy_keV"] = None
    m["n_photons_counted"] = 0
    with open(f"{b}.json", "w") as f:
        json.dump(m, f)
    msg = raises(fp.read_wavefront, b)
    is_ok &= utility.print_test(
        msg is not None and "effective_energy_keV" in msg, f"(b) N = 0: {msg[-60:]!r}"
    )

    # (c) build_field
    d = fp.read_wavefront(base)
    ok = np.array_equal(fp.build_field(d, "article"), d.real + 1j * d.imag)
    ok &= fp.build_field(d, "article").dtype == np.complex128
    is_ok &= utility.print_test(ok, "(c) article model: Re + i Im")
    phi = 0.7
    d.real[0, 0] = 3 * np.sqrt(20) * np.cos(phi)
    d.imag[0, 0] = 3 * np.sqrt(20) * np.sin(phi)
    d.energy[0, 0] = 60.0
    d.real[1, 1] = 0.0
    d.imag[1, 1] = 0.0
    d.energy[1, 1] = 0.0
    f = fp.build_field(d, "physical")
    ok = np.isclose(abs(f[0, 0]) ** 2, 60.0, rtol=1e-12)
    ok &= np.isclose(np.angle(f[0, 0]), phi, rtol=1e-12)
    is_ok &= utility.print_test(ok, f"(c) physical: |Psi_f|^2 = {abs(f[0, 0])**2:.6f}")
    ok = f[1, 1] == 0 and np.all(np.isfinite(f))
    is_ok &= utility.print_test(ok, "(c) physical: Psi = 0 gives 0, no NaN/inf")
    msg = raises(fp.build_field, d, "foo")
    is_ok &= utility.print_test(msg is not None, f"(c) unknown model: {msg!r:.60}")
    # article model: |Psi|^2 = N^2 E, physical: N E (T6)
    n, e = 7, 20.0
    d2 = fp.WavefrontData(
        real=np.full((2, 2), n * np.sqrt(e)),
        imag=np.zeros((2, 2)),
        energy=np.full((2, 2), n * e),
        spacing_mm=(1.0, 1.0),
        grid_corner_mm=(0.0, 0.0),
        plane_z_mm=0.0,
        effective_energy_keV=e,
        n_photons_counted=4 * n,
        meta={},
    )
    ia = fp.intensity(fp.build_field(d2, "article"))[0, 0]
    ip = fp.intensity(fp.build_field(d2, "physical"))[0, 0]
    ok = np.isclose(ia, n**2 * e) and np.isclose(ip, n * e)
    is_ok &= utility.print_test(ok, f"(c) T6: article {ia:.1f}, physical {ip:.1f}")

    # (d) wavelength, Fresnel number
    lam = fp.wavelength_mm(20)
    ok = np.isclose(lam, HC_KEV_MM / 20, rtol=1e-9, atol=0)
    is_ok &= utility.print_test(ok, f"(d) lambda(20 keV) = {lam:.6e} mm")
    nf = fp.fresnel_number(1e-3, lam, 100)
    ok = np.isclose(nf, (10e-3) ** 2 / (lam * 100), rtol=1e-12)
    is_ok &= utility.print_test(ok, f"(d) Fresnel number = {nf:.4g}")

    # (e) sampling
    r = fp.check_sampling(1e-4, 1.0, fp.wavelength_mm(16.5), 30.0)
    keys = {"critical_pixel_mm", "sqrt_lambda_z_mm", "message"}
    ok = r["ok"] is True and keys <= set(r)
    ok &= np.isclose(r["sqrt_lambda_z_mm"], np.sqrt(fp.wavelength_mm(16.5) * 30.0))
    is_ok &= utility.print_test(ok, f"(e) sampling ok: {r['message']}")
    r = fp.check_sampling(1e-4, 0.01, fp.wavelength_mm(16.5), 1e4)
    ok = r["ok"] is False and len(r["message"]) > 0
    is_ok &= utility.print_test(ok, f"(e) sampling not ok: {r['message']}")

    # (f) cone beam
    ok = fp.cone_beam_scaling(400, 100) == (1.25, 80.0)
    ok &= fp.cone_beam_scaling(None, 30) == (1.0, 30.0)
    is_ok &= utility.print_test(ok, "(f) cone_beam_scaling")

    # (g) intensity
    fld = np.array([[1 + 1j, 2.0], [0.0, 3j]])
    ok = np.allclose(fp.intensity(fld, M=2), np.abs(fld) ** 2 / 4)
    ok &= np.allclose(fp.intensity(fld), np.abs(fld) ** 2)
    ok &= fp.intensity(fld).dtype == np.float64
    is_ok &= utility.print_test(ok, "(g) intensity = |field|^2 / M^2")

    # (h) downsample
    img = np.arange(24, dtype=float).reshape(4, 6)
    exp = img.reshape(2, 2, 3, 2).mean(axis=(1, 3))
    ok = np.array_equal(fp.downsample(img, 2), exp)
    ok &= np.array_equal(fp.downsample(img, 1), img)
    msg = raises(fp.downsample, img, 4)
    ok &= msg is not None
    is_ok &= utility.print_test(ok, f"(h) downsample; factor 4 on 4x6: {msg!r:.60}")

    # (i) write_intensity
    wp = out / "intensity"
    fp.write_intensity(wp, img, (0.002, 0.001), (0.5, -0.25), {"model": "article"})
    back = itk.imread(f"{wp}.mhd")
    arr = itk.array_from_image(back)
    ok = np.array_equal(arr, img) and arr.dtype == np.float64
    ok &= np.allclose(list(back.GetSpacing()), [0.001, 0.002])
    ok &= np.allclose(list(back.GetOrigin()), [-0.25, 0.5])
    with open(f"{wp}.json") as f:
        js = json.load(f)
    ok &= js["meta"] == {"model": "article"}
    is_ok &= utility.print_test(ok, "(i) write_intensity round trip")

    utility.test_ok(is_ok)
