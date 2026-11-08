#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The WavefrontActor (kill_after_plane=False) only observes the transport.

A PMMA sphere (xray_phase_contrast on) is crossed by a parallel 20 keV beam.
Four runs (A, B, C and the informative D), each in its own process (Geant4
does not restart), with the same seed and the same geometry (the thin vacuum
detector is always present, only the actors change). The runs use one random
seed per event (/run/eventModulo 1 0): with the GATE default (a seed every
10000 events), killing a photon at the plane consumes fewer random numbers
and shifts the following events, so A and B would differ by statistics, not
by physics. Multithreaded mode is forced with a single worker thread (the
per-event seeding needs the MT run manager): the result is deterministic by
construction, with no floating-point summation order between threads.

  (A) WavefrontActor (kill_after_plane=False) + FluenceActor
  (B) WavefrontActor alone (kill_after_plane=True)
  (C) FluenceActor alone

Checks, all by exact equality (no tolerance):
  - real, imag and energy grids of A and B are identical;
  - the counters of A and B are identical (n_photons_counted, n_outside_grid,
    n_recrossings, n_backward_crossings and sum_energy_keV): the counters do
    not depend on the kill (checked by equality);
  - the counts and the energy of the FluenceActor are identical in A and C,
    i.e. adding the WavefrontActor does not change the transport.
A fourth run (D) is B with the default seeding; the number of wavefront pixels
that differ from B is printed as information only, it is not a check.
Sanity controls: the grid of A is not all zero and the fluence of C has
counts > 0.

The FluenceActor is not returned by sim.run(start_new_process=True), so each
run is a subprocess of this script and the results are read from the MHD
files (the fluence energy is in MeV, as written by the actor).
"""

import json
import subprocess
import sys

import itk
import numpy as np

import opengate as gate
import opengate.tests.utility as utility
from opengate.utility import g4_units

keV = g4_units.keV
mm = g4_units.mm
um = g4_units.um

RUNS = ("A", "B", "C", "D")
N_PHOTONS = 200000
PLANE_Z = 0.1 * mm
GRID = [256, 256]
SPACING = [0.5 * um, 0.5 * um]
FLUENCE_SIZE = [128, 128, 1]
FLUENCE_SPACING = [1 * um, 1 * um, 1 * um]


def simulate(paths, run):
    """One run, in the current process. Writes the MHD files in paths.output."""
    sim = gate.Simulation()
    sim.output_dir = paths.output
    sim.number_of_threads = 1
    sim.force_multithread_mode = True
    sim.visu = False
    sim.progress_bar = False
    sim.random_seed = 123456
    sim.world.size = [1 * mm, 1 * mm, 12 * mm]
    sim.world.material = "G4_Galactic"
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option1"
    sim.physics_manager.xray_phase_contrast = True
    if run != "D":
        # one random seed per event (see the module docstring)
        sim.g4_commands_after_init.append("/run/eventModulo 1 0")

    sphere = sim.add_volume("Sphere", "sphere")
    sphere.rmax = 50 * um
    sphere.translation = [0, 0, 0]
    sphere.material = "G4_PLEXIGLASS"

    det = sim.add_volume("Box", "detector")
    det.size = [FLUENCE_SIZE[0] * FLUENCE_SPACING[0],
                FLUENCE_SIZE[1] * FLUENCE_SPACING[1], 1 * um]
    det.translation = [0, 0, 5 * mm]
    det.material = "G4_Galactic"

    src = sim.add_source("GenericSource", "beam")
    src.particle = "gamma"
    src.energy.mono = 20 * keV
    src.position.type = "box"
    src.position.size = [120 * um, 120 * um, 0]
    src.position.translation = [0, 0, -1 * mm]
    src.direction.type = "momentum"
    src.direction.momentum = [0, 0, 1]
    src.number_of_primaries = N_PHOTONS

    if run in ("A", "B", "D"):
        wf = sim.add_actor("WavefrontActor", "wf")
        wf.attached_to = "world"
        wf.plane_z = PLANE_Z
        wf.plane_center = [0, 0]
        wf.size = GRID
        wf.spacing = SPACING
        wf.kill_after_plane = run != "A"
        wf.output_filename = f"wf_{run}.mhd"

    if run in ("A", "C"):
        flu = sim.add_actor("FluenceActor", "fluence")
        flu.attached_to = det
        flu.size = FLUENCE_SIZE
        flu.spacing = FLUENCE_SPACING
        flu.energy.active = True
        flu.output_filename = f"fluence_{run}.mhd"

    sim.run(start_new_process=False)


def read_mhd(path):
    return np.asarray(itk.array_from_image(itk.imread(str(path))))


def read_run(paths, run):
    """Arrays and counters of one run, read from the files on disk."""
    out = {}
    if run in ("A", "B", "D"):
        for s in ("real", "imag", "energy"):
            out[s] = read_mhd(paths.output / f"wf_{run}_{s}.mhd")
        with open(paths.output / f"wf_{run}.json") as f:
            out["json"] = json.load(f)
    if run in ("A", "C"):
        for s in ("counts", "energy"):
            out["flu_" + s] = read_mhd(paths.output / f"fluence_{run}_{s}.mhd")
    return out


def compare(name, a, b):
    """Exact equality; on failure report how many elements differ and the
    first index."""
    ok = a.shape == b.shape and np.array_equal(a, b)
    detail = f"shape {a.shape}"
    if not ok and a.shape == b.shape:
        diff = np.argwhere(a != b)
        detail = f"{len(diff)} elements differ, first index {diff[0].tolist()}"
    utility.print_test(ok, f"{name}: exactly equal ({detail})")
    return ok


if __name__ == "__main__":
    paths = utility.get_default_test_paths(__file__, output_folder="test144")

    if len(sys.argv) > 1 and sys.argv[1] in RUNS:
        simulate(paths, sys.argv[1])
        sys.exit(0)

    for run in RUNS:
        subprocess.run([sys.executable, __file__, run], check=True)
    a, b, c, d = (read_run(paths, r) for r in RUNS)

    is_ok = True
    for s in ("real", "imag", "energy"):
        is_ok &= compare(f"wavefront {s}, A vs B", a[s], b[s])

    # kill_after_plane is a configuration, not a counter: it is not compared
    ka = {k: v for k, v in a["json"].items() if k != "kill_after_plane"}
    kb = {k: v for k, v in b["json"].items() if k != "kill_after_plane"}
    ok = ka == kb
    utility.print_test(ok, f"counters, A vs B: {ka}" if ok else f"A {ka} B {kb}")
    is_ok &= ok

    is_ok &= compare("fluence counts, A vs C", a["flu_counts"], c["flu_counts"])
    is_ok &= compare("fluence energy, A vs C", a["flu_energy"], c["flu_energy"])

    ok = bool(np.any(a["real"] != 0) or np.any(a["imag"] != 0))
    ok &= a["json"]["n_photons_counted"] > 0
    utility.print_test(ok, "sanity: the wavefront grid of A is not all zero")
    is_ok &= ok
    ok = float(c["flu_counts"].sum()) > 0
    utility.print_test(
        ok, f"sanity: fluence of C has counts > 0 ({c['flu_counts'].sum()})"
    )
    is_ok &= ok

    n_diff = int(np.count_nonzero(b["energy"] != d["energy"]))
    print(f"info: default seeding (D) vs B: {n_diff} energy pixels differ")

    utility.test_ok(is_ok)
