#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Command line: Fresnel propagation of the wavefront recorded by the WavefrontActor.

Reads <base>_real.mhd, <base>_imag.mhd, <base>_energy.mhd and <base>.json, propagates
the field to one or several distances for the "article" and/or "physical" amplitude
models, for a parallel or cone beam, and writes the intensity as
<output-dir>/<base>_D<D>mm_<model>.mhd (+ .json with the meta data).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from opengate import fresnel_propagation as fp

BYTES_COMPLEX = 16
BYTES_REAL = 8


def total_ram_gb():
    """Physical RAM in GB (fallback 8 GB if it cannot be queried)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 8.0


def estimate_memory_bytes(shape, padding, pad_factor, keep_copy=True):
    """Peak memory of the propagation of one distance, in bytes.

    Arrays alive: the 3 grids read from disk (real, imag, energy: float64), the
    complex field, one copy of it (propagate works in place with padding "none";
    without the copy the field is rebuilt from the grids for each distance), the
    padded work array (only with padding), the temporary buffer that scipy may
    allocate (same size as the work array), the crop of propagate with padding,
    the temporaries of build_field / intensity (about one complex array) and the
    output image.
    """
    n = int(shape[0]) * int(shape[1])
    work = n * (int(pad_factor) ** 2 if padding != "none" else 1)
    total = 3 * n * BYTES_REAL  # grids
    total += n * BYTES_COMPLEX  # field
    if keep_copy:
        total += n * BYTES_COMPLEX  # copy that is propagated
    if padding != "none":
        total += work * BYTES_COMPLEX  # padded array
    total += work * BYTES_COMPLEX  # temporary buffer of the FFT
    if padding != "none":
        total += n * BYTES_COMPLEX  # crop returned by propagate
    total += n * BYTES_COMPLEX  # temporaries of build_field / intensity
    total += n * BYTES_REAL  # output image
    return total


def flat_field_mask(region, image_shape, pixel_mm, origin_mm):
    """Boolean mask of the flat-field region.

    region = "auto": ring of 5 % of the size on each border; otherwise the list
    [x0, x1, y0, y1] in mm (detector coordinates, relative to the optical axis).
    """
    ny, nx = image_shape
    if region == "auto":
        by, bx = max(1, int(round(0.05 * ny))), max(1, int(round(0.05 * nx)))
        mask = np.ones(image_shape, bool)
        mask[by : ny - by, bx : nx - bx] = False
        return mask
    x0, x1, y0, y1 = region
    y = origin_mm[0] + np.arange(ny) * pixel_mm[0]
    x = origin_mm[1] + np.arange(nx) * pixel_mm[1]
    my = (y >= min(y0, y1)) & (y <= max(y0, y1))
    mx = (x >= min(x0, x1)) & (x <= max(x0, x1))
    mask = my[:, None] & mx[None, :]
    if not mask.any():
        raise ValueError(f"The flat-field region {region} contains no pixel.")
    return mask


def format_distance(d):
    """Distance for the file names: fixed point with up to 6 decimals and no
    trailing zeros (never scientific notation), e.g. 50 -> '50', 12.5 -> '12.5',
    1e-5 -> '0.00001'."""
    text = f"{d:.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def read_grid_shape(base):
    """(ny, nx) from the 'size' key of <base>.json, without reading any MHD."""
    path = Path(f"{fp._base(base)}.json")
    if not path.is_file():
        raise SystemExit(f"Incomplete wavefront: file '{path}' (json) not found.")
    with open(path) as f:
        size = json.load(f).get("size")
    if not size or len(size) != 2:
        raise SystemExit(f"'{path}': missing or invalid key 'size'.")
    return int(size[1]), int(size[0])


def parse_flat_field(values):
    if values is None:
        return None
    if len(values) == 1 and values[0] == "auto":
        return "auto"
    if len(values) == 4:
        try:
            return [float(v) for v in values]
        except ValueError:
            pass
    raise SystemExit(
        "--normalize-flat-field takes 'auto' or four numbers 'x0 x1 y0 y1' (mm)."
    )


def build_parser():
    p = argparse.ArgumentParser(
        prog="opengate_fresnel_propagate",
        description="Fresnel propagation of the wavefront recorded by the "
        "WavefrontActor. Writes the intensity <output-dir>/<base>_D<D>mm_<model>"
        ".mhd and .json; D is written in fixed point without trailing zeros "
        "(50, 12.5, 0.00001).",
    )
    p.add_argument(
        "base",
        help="prefix <base> of the files of the actor (<base>_real.mhd, "
        "<base>_imag.mhd, <base>_energy.mhd, <base>.json), or <base>.mhd",
    )
    p.add_argument(
        "--distances",
        nargs="+",
        type=float,
        required=True,
        help="propagation distances D in mm (plane to detector)",
    )
    p.add_argument(
        "--model",
        choices=("article", "physical", "both"),
        default="both",
        help="amplitude model: article (Re + i Im), physical (Psi / sqrt(N), "
        "N = sum E / E_eff) or both (default)",
    )
    p.add_argument(
        "--source-distance",
        type=float,
        default=None,
        help="source to plane distance R1 in mm (cone beam, M = (R1 + D) / R1; "
        "the pixel and the origin are scaled by M, the origin relative to the "
        "optical axis x = y = 0); default: parallel beam",
    )
    p.add_argument(
        "--padding",
        choices=tuple(fp.PADDINGS),
        default="none",
        help="border treatment: none (circular, default), zero or mirror",
    )
    p.add_argument(
        "--pad-factor",
        type=int,
        default=2,
        help="grid enlargement factor with padding (default 2)",
    )
    p.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="mean over blocks of this integer factor (default 1; it must divide "
        "the grid size)",
    )
    p.add_argument(
        "--output-dir", default=".", help="output directory (default: current)"
    )
    p.add_argument(
        "--normalize-flat-field",
        nargs="+",
        metavar=("auto|X0", "X1"),
        default=None,
        help="divide the image by its mean in a flat region: 'auto' (ring of 5 %% "
        "at the border of the grid) or 'x0 x1 y0 y1' in mm, in detector "
        "coordinates (after the scaling by M)",
    )
    p.add_argument(
        "--max-memory-gb",
        type=float,
        default=None,
        help="memory limit in GB (default: 80 %% of the RAM); the command refuses "
        "to run, before allocating, a configuration that does not fit",
    )
    return p


def check_memory(shape, args, limit_bytes):
    """Return keep_copy (True if the field and its copy fit); exit with a clear
    message if even the smaller configuration does not fit."""
    for keep in (True, False):
        need = estimate_memory_bytes(shape, args.padding, args.pad_factor, keep)
        if need <= limit_bytes:
            return keep
    raise SystemExit(
        f"Not enough memory: grid {shape[0]} x {shape[1]} with padding "
        f"'{args.padding}' (factor {args.pad_factor}) needs about "
        f"{need / 1e9:.3g} GB (peak per distance), above the limit of "
        f"{limit_bytes / 1e9:.3g} GB. Use --max-memory-gb, a smaller grid, "
        "--padding none or a smaller --pad-factor."
    )


def go():
    args = build_parser().parse_args()
    flat = parse_flat_field(args.normalize_flat_field)
    if args.downsample < 1:
        raise SystemExit("--downsample must be >= 1.")
    if args.pad_factor < 1:
        raise SystemExit("--pad-factor must be >= 1.")

    # checks that need only the JSON, before any MHD is read
    shape = read_grid_shape(args.base)
    if shape[0] % args.downsample or shape[1] % args.downsample:
        raise SystemExit(
            f"--downsample {args.downsample} does not divide the grid {shape}."
        )
    limit_gb = args.max_memory_gb
    if limit_gb is None:
        limit_gb = 0.8 * total_ram_gb()
    keep_copy = check_memory(shape, args, limit_gb * 1e9)
    data = fp.read_wavefront(args.base)

    lam = fp.wavelength_mm(data.effective_energy_keV)
    pixel = data.spacing_mm
    origin = (
        data.grid_corner_mm[0] + pixel[0] / 2,
        data.grid_corner_mm[1] + pixel[1] / 2,
    )
    models = fp.MODELS if args.model == "both" else (args.model,)
    out_dir = Path(args.output_dir)
    name = fp._base(args.base).name

    print(
        f"{name}: grid {shape[0]} x {shape[1]}, pixel (y, x) = "
        f"({pixel[0]:.6g}, {pixel[1]:.6g}) mm, E = {data.effective_energy_keV:.6g} "
        f"keV, lambda = {lam:.6g} mm, N = {data.n_photons_counted}"
    )
    for model in models:
        base_field = fp.build_field(data, model) if keep_copy else None
        for d in args.distances:
            t0 = time.time()
            # propagate works in place with padding "none": use a copy
            field = base_field.copy() if keep_copy else fp.build_field(data, model)
            image, det_pixel, det_origin, meta = fp.propagate_to_detector(
                field,
                pixel,
                origin,
                lam,
                d,
                args.source_distance,
                args.padding,
                args.pad_factor,
            )
            del field
            if args.downsample > 1:
                f = args.downsample
                image = fp.downsample(image, f)
                det_origin = tuple(
                    o + (f - 1) / 2 * p for o, p in zip(det_origin, det_pixel)
                )
                det_pixel = tuple(p * f for p in det_pixel)
            if flat is not None:
                mask = flat_field_mask(flat, image.shape, det_pixel, det_origin)
                ref = float(image[mask].mean())
                if ref <= 0:
                    raise SystemExit("The flat-field mean is not positive.")
                image = image / ref
                meta["flat_field_mean"] = ref
            meta.update(
                {
                    "model": model,
                    "effective_energy_keV": data.effective_energy_keV,
                    "n_photons_counted": data.n_photons_counted,
                    "plane_z_mm": data.plane_z_mm,
                    "downsample": args.downsample,
                    "normalize_flat_field": flat,
                    "source_wavefront": str(args.base),
                }
            )
            out = out_dir / f"{name}_D{format_distance(d)}mm_{model}.mhd"
            fp.write_intensity(out, image, det_pixel, det_origin, meta)
            need = estimate_memory_bytes(
                shape, args.padding, args.pad_factor, keep_copy
            )
            print(
                f"  D = {d:g} mm, {model}: M = {meta['M']:.6g}, z_eff = "
                f"{meta['z_eff_mm']:.6g} mm, "
                f"F10 (y, x) = {meta['fresnel_number_10px']}, "
                f"{meta['sampling']['message']} time {time.time() - t0:.2f} s, "
                f"memory ~{need / 1e9:.3g} GB -> {out}"
            )
        del base_field


if __name__ == "__main__":
    go()
