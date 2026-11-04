#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helpers of the tests of opengate.fresnel_propagation (not run as a test)."""

import json

import itk
import numpy as np


def write_synthetic(
    base, ny=6, nx=4, spacing=(0.002, 0.001), energy=20.0, n=5, grids=None
):
    """Write a wavefront in the format of the WavefrontActor. spacing = (dy, dx).

    grids: optional dict with the arrays "real", "imag" and "energy" of shape
    (ny, nx); by default they are random (energy >= 0).
    """
    rng = np.random.default_rng(1)
    dy, dx = spacing
    corner = (-nx * dx / 2, -ny * dy / 2)  # (x0, y0) as in the JSON
    if grids is None:
        grids = {s: rng.normal(size=(ny, nx)) for s in ("real", "imag")}
        grids["energy"] = rng.uniform(0.0, 2.0, size=(ny, nx))  # energy is >= 0
    else:
        ny, nx = grids["real"].shape
        corner = (-nx * dx / 2, -ny * dy / 2)
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
