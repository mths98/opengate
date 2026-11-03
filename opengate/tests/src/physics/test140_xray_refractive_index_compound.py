#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import opengate.tests.utility as utility
from opengate.xray_refractive_index import (
    read_element_delta,
    interpolate_delta,
    default_energy_grid_keV,
    compound_delta,
)

# Element densities (g/cm3, NIST)
ELEMENT_DENSITY = {
    "H": 8.37480e-05,
    "C": 2.0,
    "N": 1.16528e-03,
    "O": 1.33151e-03,
    "Na": 0.971,
    "P": 2.2,
    "S": 2.0,
    "Cl": 2.99473e-03,
    "K": 0.862,
    "Ca": 1.55,
    "Al": 2.699,
}

# name: (symbols, mass fractions, density g/cm3)
TISSUES = {
    "Muscle": (
        ["H", "C", "N", "O", "Na", "P", "S", "Cl", "K"],
        [0.102, 0.143, 0.034, 0.710, 0.001, 0.002, 0.003, 0.001, 0.004],
        1.05,
    ),
    "Glandular": (
        ["H", "C", "N", "O", "Na", "P", "S", "Cl"],
        [0.106, 0.332, 0.03, 0.527, 0.001, 0.001, 0.002, 0.001],
        1.035,
    ),
    "Cancerous_mass": (
        ["H", "C", "N", "O", "Na", "P", "S", "Cl"],
        [0.106, 0.332, 0.03, 0.527, 0.001, 0.001, 0.002, 0.001],
        1.060,
    ),
    "Breast_Skin": (
        ["H", "C", "N", "O", "P", "S", "Ca", "K"],
        [0.098, 0.178, 0.050, 0.667, 0.00175, 0.00175, 0.00175, 0.00175],
        1.090,
    ),
    "Breast_Adipose": (
        ["H", "C", "N", "O", "P", "S", "Ca", "K"],
        [0.112, 0.619, 0.017, 0.251, 0.00025, 0.00025, 0.00025, 0.00025],
        0.93,
    ),
    "Breast_Glandula": (
        ["H", "C", "N", "O", "P", "S", "Ca", "K"],
        [0.102, 0.184, 0.032, 0.677, 0.00125, 0.00125, 0.00125, 0.00125],
        1.040,
    ),
}

TOLERANCE = 0.01


def eq21(symbols, w, rho, grid):
    return compound_delta(
        symbols, w, rho, [ELEMENT_DENSITY[s] for s in symbols], grid
    )


def relative_errors(symbols, w, rho, reference_name, energies):
    grid = default_energy_grid_keV()
    delta = eq21(symbols, w, rho, grid)
    ref_E, ref_d = read_element_delta(reference_name)
    mine = interpolate_delta(grid, delta, energies)
    ref = interpolate_delta(ref_E, ref_d, energies)
    return np.abs(mine / ref - 1.0)


def test_water():
    energies = np.array([10, 16.5, 20, 30, 60, 100])
    ok = True
    for name in ["Water", "H2O", "Water_new"]:
        err = relative_errors(["H", "O"], [0.1119, 0.8881], 1.0, name, energies)
        ok = ok and bool(np.all(err <= TOLERANCE))
        utility.print_test(
            np.all(err <= TOLERANCE),
            f"water vs {name}: max rel. error {err.max():.3e} "
            f"({', '.join(f'{x:.2e}' for x in err)})",
        )
    return ok


def test_pmma():
    energies = np.array([5, 10, 20, 30])
    err = relative_errors(
        ["C", "H", "O"], [0.59985, 0.08054, 0.31961], 1.19, "PMMA", energies
    )
    ok = bool(np.all(err <= TOLERANCE))
    utility.print_test(
        ok,
        f"PMMA vs PMMA.txt: max rel. error {err.max():.3e} "
        f"({', '.join(f'{x:.2e}' for x in err)})",
    )
    return ok


def test_tissues():
    ok = True
    for name, (symbols, w, rho) in TISSUES.items():
        ref_E, _ = read_element_delta(name)
        energies = np.geomspace(max(5.0, ref_E[0]), min(100.0, ref_E[-1]), 8)
        err = relative_errors(symbols, w, rho, name, energies)
        this_ok = bool(np.all(err <= TOLERANCE))
        ok = ok and this_ok
        utility.print_test(
            this_ok,
            f"{name}: max rel. error {err.max():.3e} "
            f"({', '.join(f'{x:.2e}' for x in err)})",
        )
    return ok


def test_single_element():
    grid = default_energy_grid_keV()
    res = compound_delta(["Al"], [1.0], 2.699, [2.699], grid)
    E, d = read_element_delta("Al")
    ref = interpolate_delta(E, d, grid)
    ok = bool(np.allclose(res, ref, rtol=1e-12, atol=0))
    utility.print_test(ok, "N = 1: compound of Al equals Al.txt on the grid")
    return ok


def gamos_water_delta(e):
    """Recipe of the original code: linear interpolation of each element on a
    250-point linear grid, mixture sum(delta_i * w_i * rho / rho_i), then
    linear interpolation on that grid."""
    grid = np.linspace(1.0, 150.0, 250)
    w = {"H": 0.1119, "O": 0.8881}
    rho = 1.0
    total = np.zeros_like(grid)
    for s, ws in w.items():
        E, d = read_element_delta(s)
        total += np.interp(grid, E, d) * ws * rho / ELEMENT_DENSITY[s]
    return np.interp(e, grid, total)


def test_gamos():
    energies = np.array([5, 10, 16.5, 20, 30, 60, 100, 140])
    grid = default_energy_grid_keV()
    mine = interpolate_delta(
        grid, eq21(["H", "O"], [0.1119, 0.8881], 1.0, grid), energies
    )
    gamos = gamos_water_delta(energies)
    diff = np.abs(mine / gamos - 1.0)
    ok = bool(np.all(diff < 0.005))
    utility.print_test(
        ok,
        f"water vs GAMOS recipe: max rel. diff {diff.max():.3e} "
        f"({', '.join(f'{x:.2e}' for x in diff)})",
    )
    return ok


if __name__ == "__main__":
    is_ok = True
    is_ok = test_water() and is_ok
    is_ok = test_pmma() and is_ok
    is_ok = test_tissues() and is_ok
    is_ok = test_single_element() and is_ok
    is_ok = test_gamos() and is_ok
    utility.test_ok(is_ok)
