#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Basic tests of the WavefrontActor (complex sum on a virtual plane).

The actor accumulates the phase Phi = sum of -2 pi delta L / lambda of each
photon and, at the first crossing of the plane z = plane_z, adds
sqrt(E/keV) exp(i Phi) and E/keV to the pixel hit by the photon.

The gamma interactions (GammaGeneralProc) are inactivated: every photon goes
straight through the slabs, so that all the photons of a pixel have the same
phase and the expected values are exact. The refractive index decrement
delta is computed as apply_xray_rindex does it (compound_delta on the
default energy grid, RINDEX = 1 - delta, then the linear interpolation of
G4PhysicsVector::Value).
"""

import json
import subprocess
import sys

import itk
import numpy as np

import opengate as gate
import opengate.tests.utility as utility
from opengate.utility import g4_units
from opengate.xray_refractive_index import (
    compound_delta,
    default_energy_grid_keV,
    read_reference_densities,
)

keV = g4_units.keV
mm = g4_units.mm
um = g4_units.um

# h c in Geant4 units, with the same expressions as CLHEP
H_PLANCK = 6.62607015e-34 * g4_units.joule * g4_units.s
C_LIGHT = 2.99792458e8 * g4_units.m / g4_units.s

PLANE_Z = 100 * mm
GRID = [10, 10]
SPACING = [0.1 * mm, 0.1 * mm]
N_PHOTONS = 200
MATERIALS = ("G4_Galactic", "G4_PLEXIGLASS", "G4_Al")

# setups placed side by side along x, all at z < PLANE_Z (except (c))
X_VACUUM = 0 * mm
X_SLAB = 20 * mm
X_TWO_SLABS = 40 * mm
X_THIN = 60 * mm
X_HALF = 80 * mm
T_SLAB = 1 * mm
T_AL = 0.5 * mm
T_THIN = 10 * um
T_BLOCK = 2 * mm
JSON_KEYS = {
    "n_photons_counted",
    "n_outside_grid",
    "n_recrossings",
    "effective_energy_keV",
    "plane_z_mm",
    "size",
    "spacing_mm",
    "kill_after_plane",
}


def log_materials(simulation_engine):
    """Hook called after the initialization: record the composition of the
    materials and the RINDEX values stored in their property tables."""
    import opengate_core as g4

    g_cm3 = g4_units.g / g4_units.cm3
    nist = g4.G4NistManager.Instance()
    log = {}
    for name in MATERIALS:
        mat = nist.FindOrBuildMaterial(name)
        n = mat.GetNumberOfElements()
        symbols = [str(mat.GetElement(i).GetSymbol()) for i in range(n)]
        mpt = mat.GetMaterialPropertiesTable()
        stored = None if mpt is None else mpt.GetPropertyValues("RINDEX")
        log[name] = {
            "symbols": symbols,
            "fractions": [mat.GetElementFraction(i) for i in range(n)],
            "density": mat.GetDensity() / g_cm3,
            "pure_density": [
                nist.FindOrBuildMaterial("G4_" + s).GetDensity() / g_cm3
                for s in symbols
            ],
            "rindex": None if stored is None else [list(stored[0]), list(stored[1])],
        }
    simulation_engine.user_hook_log["materials"] = log


def remove_vacuum_rindex(simulation_engine):
    """Hook called after the initialization: remove the property table (and
    thus RINDEX) of the world material."""
    import opengate_core as g4

    for mat in g4.G4Material.GetMaterialTable:
        if str(mat.GetName()) == "G4_Galactic":
            mat.SetMaterialPropertiesTable(None)


def rindex_table(composition):
    """(energies, n) as written by apply_xray_rindex."""
    overrides = read_reference_densities(None)
    rho_i = [
        overrides.get(s, rho)
        for s, rho in zip(composition["symbols"], composition["pure_density"])
    ]
    grid = np.asarray(default_energy_grid_keV(), dtype=float)
    delta = compound_delta(
        composition["symbols"],
        composition["fractions"],
        composition["density"],
        rho_i,
        grid,
        None,
    )
    return list(grid * keV), list(1.0 - delta)


def g4_linear_value(energies, values, e):
    """G4PhysicsVector::Value (linear interpolation, no spline)."""
    if e <= energies[0]:
        return values[0]
    if e >= energies[-1]:
        return values[-1]
    idx = int(np.searchsorted(energies, e, side="right")) - 1
    x1 = energies[idx]
    dl = energies[idx + 1] - x1
    y1 = values[idx]
    dy = values[idx + 1] - y1
    b = (e - x1) / dl
    return y1 + b * dy


def delta_at(composition, energy):
    """delta = 1 - RINDEX(E), as seen by the actor."""
    energies, values = rindex_table(composition)
    return 1.0 - g4_linear_value(energies, values, energy)


def phase(composition, energy, thickness):
    """Phase -2 pi delta T / lambda, with lambda = h c / E."""
    lam = H_PLANCK * C_LIGHT / energy
    return -2.0 * np.pi * delta_at(composition, energy) * thickness / lam


def add_disc_source(sim, name, x, energy, n=N_PHOTONS):
    source = sim.add_source("GenericSource", name)
    source.particle = "gamma"
    source.energy.mono = energy
    source.position.type = "disc"
    source.position.radius = 0.4 * mm
    source.position.translation = [x, 0, 0]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.number_of_primaries = n
    return source


def add_point_source(sim, name, x, y, n):
    source = sim.add_source("GenericSource", name)
    source.particle = "gamma"
    source.energy.mono = 20 * keV
    source.position.type = "point"
    source.position.translation = [x, y, 0]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.number_of_primaries = n
    return source


def add_slab(sim, name, x, z, thickness, material="G4_PLEXIGLASS"):
    slab = sim.add_volume("Box", name)
    slab.size = [10 * mm, 10 * mm, thickness]
    slab.translation = [x, 0, z]
    slab.material = material
    return slab


def add_wavefront(sim, name, x, plane_z=PLANE_Z):
    wf = sim.add_actor("WavefrontActor", name)
    wf.attached_to = "world"
    wf.plane_z = plane_z
    wf.plane_center = [x, 0]
    wf.size = GRID
    wf.spacing = SPACING
    wf.output_filename = f"{name}.mhd"
    return wf


def make_base_sim(paths, phase_contrast=True, hook=log_materials):
    sim = gate.Simulation()
    sim.output_dir = paths.output
    sim.number_of_threads = 1
    sim.visu = False
    sim.random_seed = 123456
    sim.world.size = [300 * mm, 50 * mm, 400 * mm]
    sim.world.material = "G4_Galactic"
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option1"
    sim.physics_manager.xray_phase_contrast = phase_contrast
    # straight photons: no absorption and no scattering in the slabs. In
    # option1, phot and compt are sub-processes of GammaGeneralProc
    sim.g4_commands_after_init.append("/process/inactivate GammaGeneralProc")
    sim.user_hook_after_init = hook
    return sim


def make_main_sim(paths):
    """20 keV: (a) vacuum, (b) PMMA slab, (b) PMMA + Al, (b2) thin PMMA,
    (c) plane in the middle of a PMMA block, focus 3 (unreachable plane)."""
    sim = make_base_sim(paths)
    e = 20 * keV
    add_disc_source(sim, "src_vacuum", X_VACUUM, e)
    add_wavefront(sim, "wf_vacuum", X_VACUUM)

    add_slab(sim, "slab", X_SLAB, 50 * mm, T_SLAB)
    add_disc_source(sim, "src_slab", X_SLAB, e)
    add_wavefront(sim, "wf_slab", X_SLAB)

    add_slab(sim, "two_pmma", X_TWO_SLABS, 50 * mm, T_SLAB)
    add_slab(sim, "two_al", X_TWO_SLABS, 60 * mm, T_AL, "G4_Al")
    add_disc_source(sim, "src_two", X_TWO_SLABS, e)
    add_wavefront(sim, "wf_two", X_TWO_SLABS)

    add_slab(sim, "thin", X_THIN, 50 * mm, T_THIN)
    add_disc_source(sim, "src_thin", X_THIN, e)
    add_wavefront(sim, "wf_thin", X_THIN)

    add_slab(sim, "block", X_HALF, PLANE_Z, T_BLOCK)
    add_disc_source(sim, "src_half", X_HALF, e)
    add_wavefront(sim, "wf_half", X_HALF)

    # focus 3: plane beyond the world (half length 200 mm)
    add_wavefront(sim, "wf_far", X_VACUUM, plane_z=500 * mm)
    return sim


def make_energy_sim(paths, energy):
    """(b3) thin PMMA slab at another energy."""
    sim = make_base_sim(paths)
    add_slab(sim, "thin", X_THIN, 50 * mm, T_THIN)
    add_disc_source(sim, "src_thin", X_THIN, energy)
    add_wavefront(sim, "wf_thin", X_THIN)
    return sim


def make_pixel_sim(paths):
    """(f) pencil beams: one inside the grid, one outside."""
    sim = make_base_sim(paths)
    add_point_source(sim, "beam_in", 0.25 * mm, -0.15 * mm, 50)
    add_point_source(sim, "beam_out", 0.7 * mm, 0, 30)
    add_wavefront(sim, "wf_pixel", 0)
    return sim


def counted(wf):
    """Pixels with at least one photon."""
    return wf.energy > 0


def report(ok, name, measured, expected, error, tol):
    utility.print_test(
        ok,
        f"{name}: measured {measured} expected {expected} "
        f"error {error:.3e} (tol {tol:.0e})",
    )
    return ok


def phase_residual(psi, expected):
    """Angle between psi/|psi| and exp(i expected), without 2 pi folding:
    (largest absolute residual, measured phase = expected + mean residual)."""
    res = np.angle(psi * np.exp(-1j * expected))
    return float(np.max(np.abs(res))), f"{expected + float(np.mean(res)):.12f}"


def check_rindex(log):
    """The RINDEX recomputed here is exactly the one stored in Geant4."""
    ok = True
    for name in MATERIALS:
        stored = log[name]["rindex"]
        energies, values = rindex_table(log[name])
        ok &= stored is not None and stored[0] == energies and stored[1] == values
    utility.print_test(ok, "RINDEX recomputed = RINDEX stored in Geant4")
    return ok


def check_outputs(paths, wf):
    """The four files exist, with spacing, origin and JSON keys."""
    base = paths.output / wf.name
    files = [
        base.parent / f"{base.name}_{s}.mhd" for s in ("real", "imag", "energy")
    ]
    json_file = base.parent / f"{base.name}.json"
    ok = all(f.is_file() for f in files) and json_file.is_file()
    if not ok:
        utility.print_test(False, f"outputs of {wf.name}: missing files")
        return False
    x0 = wf.plane_center[0] - GRID[0] * SPACING[0] / 2
    y0 = wf.plane_center[1] - GRID[1] * SPACING[1] / 2
    # ITK origin = centre of the first pixel
    origin = [x0 + SPACING[0] / 2, y0 + SPACING[1] / 2]
    arrays = [wf.real, wf.imag, wf.energy]
    for f, arr in zip(files, arrays):
        img = itk.imread(str(f))
        data = itk.array_from_image(img)
        ok &= img.GetImageDimension() == 2
        ok &= np.allclose(list(img.GetSpacing()), SPACING, rtol=0, atol=1e-12)
        ok &= np.allclose(list(img.GetOrigin()), origin, rtol=0, atol=1e-12)
        ok &= data.shape == (GRID[1], GRID[0]) and np.array_equal(data, arr)
    with open(json_file) as f:
        meta = json.load(f)
    ok &= set(meta.keys()) == JSON_KEYS
    ok &= meta["n_photons_counted"] == wf.counters["n_photons_counted"]
    ok &= meta["size"] == GRID and meta["plane_z_mm"] == PLANE_Z / mm
    ok &= np.allclose(meta["spacing_mm"], [s / mm for s in SPACING])
    utility.print_test(
        ok, f"outputs of {wf.name}: 3 MHD (2D, spacing, origin) + JSON keys"
    )
    return ok


if __name__ == "__main__":
    paths = utility.get_default_test_paths(__file__, output_folder="test142_basic")

    # child scenarios, launched by the main test (see run_child below)
    if len(sys.argv) > 1 and sys.argv[1] == "--no-phase-contrast":
        sim = make_base_sim(paths, phase_contrast=False)
        add_disc_source(sim, "src", 0, 20 * keV, 5)
        add_wavefront(sim, "wf", 0)
        sim.run(start_new_process=False)
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--no-rindex":
        sim = make_base_sim(paths, hook=remove_vacuum_rindex)
        add_disc_source(sim, "src", 0, 20 * keV, 5)
        add_wavefront(sim, "wf", 0)
        sim.run(start_new_process=False)
        sys.exit(0)

    def run_child(option):
        res = subprocess.run(
            [sys.executable, __file__, option], capture_output=True, text=True
        )
        return res.returncode, res.stdout + res.stderr

    tol = 1e-9
    is_ok = True

    # ---- main simulation at 20 keV
    sim = make_main_sim(paths)
    # Geant4 state persists within a process: each run in its own
    sim.run(start_new_process=True)
    comp = sim.user_hook_log["materials"]
    is_ok &= check_rindex(comp)
    pmma, al = comp["G4_PLEXIGLASS"], comp["G4_Al"]
    get = sim.actor_manager.get_actor
    e20 = 20 * keV

    # (a) vacuum: no phase, |psi| = n sqrt(20), sum E = 20 n
    wf = get("wf_vacuum")
    m = counted(wf)
    n_pix = wf.energy / 20.0
    psi = wf.real + 1j * wf.imag
    err_phase = float(np.max(np.abs(np.angle(psi[m]))))
    err_amp = float(np.max(np.abs(np.abs(psi[m]) - np.round(n_pix[m]) * np.sqrt(20))))
    err_n = float(np.max(np.abs(n_pix - np.round(n_pix))))
    n = wf.counters["n_photons_counted"]
    ok = err_phase < tol and err_amp < 1e-9 and err_n < 1e-12 and n == N_PHOTONS
    ok &= int(np.round(n_pix.sum())) == N_PHOTONS
    is_ok &= report(ok, f"(a) vacuum, N = {n}, |Phi|", 0, 0, err_phase, tol)
    is_ok &= report(
        ok, "(a) vacuum, |psi| - n_pix sqrt(20)", 0, 0, max(err_amp, err_n), 1e-9
    )
    ok = wf.counters["n_outside_grid"] == 4 * N_PHOTONS
    ok &= wf.counters["n_recrossings"] == 0
    is_ok &= report(
        ok,
        "(a) vacuum, photons of the other setups outside the grid",
        wf.counters["n_outside_grid"],
        4 * N_PHOTONS,
        0,
        0,
    )
    is_ok &= check_outputs(paths, wf)

    # (b) PMMA slab, T = 1 mm
    expected = phase(pmma, e20, T_SLAB)
    wf = get("wf_slab")
    psi = (wf.real + 1j * wf.imag)[counted(wf)]
    err, measured = phase_residual(psi, expected)
    n = wf.counters["n_photons_counted"]
    ok = err < tol and n == N_PHOTONS
    is_ok &= report(
        ok, f"(b) PMMA 1 mm, N = {n}, Phi", measured, f"{expected:.12f}", err, tol
    )

    # (b) PMMA 1 mm + Al 0.5 mm
    expected = phase(pmma, e20, T_SLAB) + phase(al, e20, T_AL)
    wf = get("wf_two")
    psi = (wf.real + 1j * wf.imag)[counted(wf)]
    err, measured = phase_residual(psi, expected)
    n = wf.counters["n_photons_counted"]
    ok = err < tol and n == N_PHOTONS
    is_ok &= report(
        ok,
        f"(b) PMMA 1 mm + Al 0.5 mm, N = {n}, Phi",
        measured,
        f"{expected:.12f}",
        err,
        tol,
    )

    # (b2) PMMA 10 um: |Phi| < pi, compared without folding
    phi_thin = {}
    expected = phase(pmma, e20, T_THIN)
    wf = get("wf_thin")
    measured = np.angle((wf.real + 1j * wf.imag)[counted(wf)])
    err = float(np.max(np.abs(measured - expected)))
    phi_thin[20] = float(np.mean(measured))
    ok = err < tol and abs(expected) < np.pi
    ok &= wf.counters["n_photons_counted"] == N_PHOTONS
    is_ok &= report(
        ok,
        "(b2) PMMA 10 um, 20 keV, angle(psi)",
        f"{phi_thin[20]:.12f}",
        f"{expected:.12f}",
        err,
        tol,
    )

    # (c) plane in the middle of a 2 mm PMMA block: phase of 1 mm
    expected = phase(pmma, e20, T_BLOCK / 2)
    wf = get("wf_half")
    psi = (wf.real + 1j * wf.imag)[counted(wf)]
    err, measured = phase_residual(psi, expected)
    n = wf.counters["n_photons_counted"]
    ok = err < tol and n == N_PHOTONS
    is_ok &= report(
        ok,
        f"(c) plane in a 2 mm block, N = {n}, Phi",
        measured,
        f"{expected:.12f}",
        err,
        tol,
    )

    # focus 3: plane never reached
    wf = get("wf_far")
    c = wf.counters
    ok = c["n_photons_counted"] == 0 and c["n_outside_grid"] == 0
    ok &= c["n_recrossings"] == 0 and not np.any(wf.energy)
    utility.print_test(ok, f"(focus 3) plane beyond the world: counters {c}")
    is_ok &= ok

    # ---- (b3) energy dependence of the thin slab phase
    for e_kev in (10, 40):
        sim_e = make_energy_sim(paths, e_kev * keV)
        sim_e.run(start_new_process=True)
        expected = phase(pmma, e_kev * keV, T_THIN)
        wf = sim_e.actor_manager.get_actor("wf_thin")
        measured = np.angle((wf.real + 1j * wf.imag)[counted(wf)])
        err = float(np.max(np.abs(measured - expected)))
        phi_thin[e_kev] = float(np.mean(measured))
        ok = err < tol and wf.counters["n_photons_counted"] == N_PHOTONS
        ok &= abs(wf.counters["sum_energy_keV"] - e_kev * N_PHOTONS) < 1e-9
        is_ok &= report(
            ok,
            f"(b3) PMMA 10 um, {e_kev} keV, angle(psi)",
            f"{phi_thin[e_kev]:.12f}",
            f"{expected:.12f}",
            err,
            tol,
        )
    ratio = phi_thin[10] / phi_thin[40]
    ratio_expected = (delta_at(pmma, 10 * keV) * 10) / (delta_at(pmma, 40 * keV) * 40)
    err = abs(ratio / ratio_expected - 1)
    is_ok &= report(
        err < tol,
        "(b3) Phi(10 keV) / Phi(40 keV), relative",
        f"{ratio:.12f}",
        f"{ratio_expected:.12f}",
        err,
        tol,
    )

    # ---- (h) no xray_phase_contrast: exception citing the option
    code, out = run_child("--no-phase-contrast")
    ok = code != 0 and "xray_phase_contrast" in out and "WavefrontActor" in out
    utility.print_test(ok, f"(h) no xray_phase_contrast: exit code {code}")
    if not ok:
        print(out[-2000:])
    is_ok &= ok

    # ---- material without RINDEX: Geant4 fatal exception
    code, out = run_child("--no-rindex")
    ok = code != 0 and "WavefrontActor01" in out
    utility.print_test(ok, f"(i) no RINDEX: exit code {code}, WavefrontActor01")
    if not ok:
        print(out[-2000:])
    is_ok &= ok

    # ---- (f) pixel index, in the main process (arrays read without copy)
    sim_f = make_pixel_sim(paths)
    sim_f.run(start_new_process=False)
    wf = sim_f.actor_manager.get_actor("wf_pixel")
    hit = np.argwhere(wf.energy > 0)
    c = wf.counters
    ok = hit.tolist() == [[3, 7]] and c["n_photons_counted"] == 50
    ok &= abs(wf.energy[3, 7] - 20 * 50) < 1e-9 and c["n_outside_grid"] == 30
    utility.print_test(
        ok,
        f"(f) pencil beam at (0.25, -0.15) mm: pixels hit (j, i) = {hit.tolist()}, "
        f"expected [[3, 7]]; x = 0.7 mm -> n_outside_grid = {c['n_outside_grid']}",
    )
    is_ok &= ok
    view = wf.real.base is not None and not wf.real.flags.owndata
    utility.print_test(view, "(f) actor.real is a view on the C++ grid (no copy)")
    is_ok &= view
    is_ok &= check_outputs(paths, wf)

    utility.test_ok(is_ok)
