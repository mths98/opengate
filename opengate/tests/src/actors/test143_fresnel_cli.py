#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests of the command line opengate_fresnel_propagate (no simulation).

A synthetic wavefront (a Gaussian phase on a 256 x 256 grid) is written in the
exact format of the WavefrontActor and go() is called with a built sys.argv.
"""

import json
import sys
from pathlib import Path

import itk
import numpy as np

import opengate.tests.utility as utility
from opengate import fresnel_propagation as fp
from opengate.bin.opengate_fresnel_propagate import go
from test143_fresnel_helpers import write_synthetic

N, PIX, ENERGY = 256, 1e-3, 20.0


def run(args):
    old = sys.argv
    sys.argv = ["opengate_fresnel_propagate"] + [str(a) for a in args]
    try:
        go()
        return None
    except SystemExit as e:
        return e
    finally:
        sys.argv = old


def read_out(path):
    img = itk.imread(str(path))
    with open(Path(path).with_suffix(".json")) as f:
        return np.array(itk.array_from_image(img)), img, json.load(f)


if __name__ == "__main__":
    paths = utility.get_default_test_paths(__file__, output_folder="test143_cli")
    out = Path(paths.output)
    out.mkdir(parents=True, exist_ok=True)
    is_ok = True

    c = (np.arange(N) - N // 2) * PIX
    r2 = c[:, None] ** 2 + c[None, :] ** 2
    phase = 1.5 * np.exp(-r2 / 0.04**2)
    amp = 1.0 + 0.2 * np.exp(-r2 / 0.06**2)
    psi = amp * np.exp(1j * phase)
    grids = {"real": psi.real, "imag": psi.imag, "energy": np.abs(psi) ** 2 * 7.0}
    base = out / "wave"
    write_synthetic(str(base), spacing=(PIX, PIX), energy=ENERGY, n=7, grids=grids)
    lam = fp.wavelength_mm(ENERGY)
    dists = (50.0, 200.0)

    # two distances and two models: 4 files equal to propagate + intensity
    od = out / "res"
    r = run([base, "--distances", *dists, "--model", "both", "--output-dir", od])
    is_ok &= utility.print_test(r is None, "run with two distances and both models")
    data = fp.read_wavefront(base)
    ok = True
    err = 0.0
    for d in dists:
        for model in ("article", "physical"):
            f = od / f"wave_D{d:g}mm_{model}.mhd"
            if not (f.is_file() and f.with_suffix(".json").is_file()):
                ok = False
                continue
            img, _, js = read_out(f)
            ref = fp.intensity(
                fp.propagate(fp.build_field(data, model), (PIX, PIX), lam, d)
            )
            err = max(err, np.max(np.abs(img - ref)))
            ok &= js["meta"]["M"] == 1.0 and js["meta"]["model"] == model
    is_ok &= utility.print_test(ok and err < 1e-12, f"4 outputs, max err. {err:.1e}")

    # cone beam
    od = out / "cone"
    run([base, "--distances", 100, "--model", "article", "--source-distance", 400,
         "--output-dir", od])
    img, itk_img, js = read_out(od / "wave_D100mm_article.mhd")
    sp = np.array(itk_img.GetSpacing())
    org = np.array(itk_img.GetOrigin())
    org_plane = -N * PIX / 2 + PIX / 2
    ok = abs(js["meta"]["M"] - 1.25) < 1e-15
    ok &= abs(js["meta"]["z_eff_mm"] - 80) < 1e-12
    ok &= np.allclose(sp, PIX * 1.25, rtol=1e-12)
    ok &= np.allclose(org, org_plane * 1.25, rtol=1e-12)
    ref = fp.intensity(fp.propagate(fp.build_field(data, "article"), (PIX, PIX),
                                    lam, 80.0), 1.25)
    ok &= np.max(np.abs(img - ref)) < 1e-12
    is_ok &= utility.print_test(
        ok, f"cone beam: M = {js['meta']['M']}, pixel = {sp[0]:.6g} mm"
    )

    # downsample
    od = out / "down"
    run([base, "--distances", 100, "--model", "article", "--downsample", 4,
         "--output-dir", od])
    img, itk_img, js = read_out(od / "wave_D100mm_article.mhd")
    ok = img.shape == (64, 64) and np.allclose(itk_img.GetSpacing(), 4 * PIX)
    is_ok &= utility.print_test(ok, f"--downsample 4: shape {img.shape}")

    # flat field on a wave without object
    flat = {"real": np.ones((N, N)), "imag": np.zeros((N, N)),
            "energy": np.ones((N, N))}
    fbase = out / "flat"
    write_synthetic(str(fbase), spacing=(PIX, PIX), energy=ENERGY, n=1, grids=flat)
    od = out / "flatres"
    run([fbase, "--distances", 100, "--model", "article", "--normalize-flat-field",
         "auto", "--output-dir", od])
    img, _, _ = read_out(od / "flat_D100mm_article.mhd")
    err = np.max(np.abs(img - 1))
    is_ok &= utility.print_test(err < 1e-12, f"flat field auto: {err:.1e}")
    # the same with the interval in mm, on the wave with an object (mean = 1)
    od = out / "flatbox"
    run([base, "--distances", 100, "--model", "article", "--normalize-flat-field",
         -0.12, 0.12, -0.12, 0.12, "--output-dir", od])
    img, itk_img, js = read_out(od / "wave_D100mm_article.mhd")
    cc = itk_img.GetOrigin()[0] + np.arange(N) * itk_img.GetSpacing()[0]
    box = np.abs(cc) <= 0.12
    mean = img[np.ix_(box, box)].mean()
    is_ok &= utility.print_test(abs(mean - 1) < 1e-12, f"flat field box: {mean:.12f}")

    # memory: refuses before allocating
    od = out / "mem"
    r = run([base, "--distances", 100, "--max-memory-gb", 0.0001,
             "--output-dir", od])
    msg = str(r.code) if r is not None else ""
    ok = r is not None and r.code not in (0, None) and "memory" in msg
    ok &= not od.exists() or not list(od.glob("*.mhd"))
    is_ok &= utility.print_test(ok, f"memory limit: {msg[:70]!r}")

    # the refusal comes before any MHD is read: JSON declaring 10000^2, no MHD
    big = out / "big"
    write_synthetic(str(big), spacing=(PIX, PIX), energy=ENERGY, n=1)
    for suffix in ("real", "imag", "energy"):
        Path(f"{big}_{suffix}.mhd").unlink()
    with open(f"{big}.json") as f:
        js = json.load(f)
    js["size"] = [10000, 10000]
    with open(f"{big}.json", "w") as f:
        json.dump(js, f)
    r = run([big, "--distances", 100, "--max-memory-gb", 1, "--output-dir", out])
    msg = str(r.code) if r is not None else ""
    ok = "memory" in msg and "not found" not in msg
    is_ok &= utility.print_test(ok, f"refusal before reading the grids: {msg[:50]!r}")
    r = run([big, "--distances", 100, "--downsample", 3, "--output-dir", out])
    msg = str(r.code) if r is not None else ""
    is_ok &= utility.print_test("downsample" in msg, f"downsample check: {msg[:50]!r}")
    # file name without loss or scientific notation
    from opengate.bin.opengate_fresnel_propagate import format_distance

    names = [format_distance(v) for v in (50.0, 12.5, 1e-5, 0.0, 1234567.0)]
    is_ok &= utility.print_test(
        names == ["50", "12.5", "0.00001", "0", "1234567"], f"file names {names}"
    )

    # D = 0: Parseval, the output is |Psi|^2
    od = out / "d0"
    run([base, "--distances", 0, "--model", "article", "--output-dir", od])
    img, _, _ = read_out(od / "wave_D0mm_article.mhd")
    err = np.max(np.abs(img - np.abs(psi) ** 2))
    is_ok &= utility.print_test(err < 1e-12, f"D = 0: |Psi|^2, err. {err:.1e}")

    utility.test_ok(is_ok)
