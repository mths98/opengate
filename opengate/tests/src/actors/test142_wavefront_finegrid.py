#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fine grid test of the WavefrontActor: 10000 x 10000 pixels of 0.1 um.

The three double grids (real, imag, energy) hold 3 x 800 MB = 2.4 GB. The
test checks that:
- the peak resident memory of the process stays below 4 GB, i.e. the grids
  are exposed to Python as views on the C++ buffers (no copy);
- actor.real, actor.imag and actor.energy do not own their data;
- the MHD files written on disk, read back, are identical to the arrays.

The simulation runs in the main process (start_new_process=False), where the
grids are used. With start_new_process=True the arrays would be pickled to the
parent process and the memory would be duplicated: this is not the use case
tested here. Run this test alone: it is heavy (memory, disk and time).
"""

import resource
import time

import itk
import numpy as np

import opengate as gate
import opengate.tests.utility as utility
from opengate.utility import g4_units

keV = g4_units.keV
mm = g4_units.mm
um = g4_units.um

N_PIXELS = 10000
SPACING = 0.1 * um
N_PHOTONS = 100000
MAX_RSS_GB = 4.0

if __name__ == "__main__":
    paths = utility.get_default_test_paths(__file__, output_folder="test142_finegrid")
    t0 = time.time()

    sim = gate.Simulation()
    sim.output_dir = paths.output
    sim.number_of_threads = 4
    sim.visu = False
    sim.random_seed = 123456
    sim.world.size = [10 * mm, 10 * mm, 400 * mm]
    sim.world.material = "G4_Galactic"
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option1"
    sim.physics_manager.xray_phase_contrast = True

    # photons in vacuum, inside the 1 mm x 1 mm grid
    source = sim.add_source("GenericSource", "src")
    source.particle = "gamma"
    source.energy.mono = 20 * keV
    source.position.type = "disc"
    source.position.radius = 0.4 * mm
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.number_of_primaries = N_PHOTONS

    wf = sim.add_actor("WavefrontActor", "wf")
    wf.attached_to = "world"
    wf.plane_z = 100 * mm
    wf.size = [N_PIXELS, N_PIXELS]
    wf.spacing = [SPACING, SPACING]
    wf.output_filename = "wf.mhd"

    sim.run(start_new_process=False)
    t_sim = time.time() - t0

    # ru_maxrss is in KB on Linux
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    grids_gb = 3 * N_PIXELS * N_PIXELS * 8 / 1024**3
    print(f"Simulation + output: {t_sim:.1f} s")
    print(f"Peak memory (ru_maxrss): {peak_gb:.2f} GB, grids {grids_gb:.2f} GB")

    is_ok = True
    ok = peak_gb < MAX_RSS_GB
    utility.print_test(ok, f"peak memory {peak_gb:.2f} GB < {MAX_RSS_GB} GB")
    is_ok &= ok

    wf = sim.actor_manager.get_actor("wf")
    arrays = {"real": wf.real, "imag": wf.imag, "energy": wf.energy}
    ok = True
    for name, arr in arrays.items():
        ok &= arr.shape == (N_PIXELS, N_PIXELS) and arr.dtype == np.float64
        ok &= arr.base is not None and not arr.flags.owndata
    utility.print_test(ok, "real, imag, energy: (10000, 10000) float64 views, no copy")
    is_ok &= ok

    n = wf.counters["n_photons_counted"]
    ok = n == N_PHOTONS and wf.counters["n_outside_grid"] == 0
    ok &= abs(float(wf.energy.sum()) - 20.0 * N_PHOTONS) < 1e-6
    utility.print_test(ok, f"N = {n} photons counted, none outside the grid")
    is_ok &= ok

    # MHD read back: identical to the arrays
    base = paths.output / "wf"
    files = [paths.output / f"wf_{s}.mhd" for s in arrays]
    for name, f in zip(arrays, files):
        img = itk.imread(str(f))
        data = itk.array_from_image(img)
        ok = data.shape == (N_PIXELS, N_PIXELS)
        spacing = list(img.GetSpacing())
        ok &= np.allclose(spacing, [SPACING, SPACING], rtol=0, atol=1e-12)
        ok &= np.array_equal(data, arrays[name])
        del data, img
        utility.print_test(ok, f"MHD {f.name} read back == actor.{name}")
        is_ok &= ok

    # clean the 2.4 GB written on disk
    for f in files:
        f.unlink(missing_ok=True)
        f.with_suffix(".raw").unlink(missing_ok=True)
    base.with_suffix(".json").unlink(missing_ok=True)
    print(f"Total time: {time.time() - t0:.1f} s")

    utility.test_ok(is_ok)
