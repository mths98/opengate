#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys

import opengate as gate
import opengate.tests.utility as utility
from opengate.utility import g4_units

keV = g4_units.keV
cm = g4_units.cm


def log_processes(simulation_engine):
    """Hook called after the initialization: record the process names
    attached to the gamma."""
    import opengate_core as g4

    gamma = g4.G4ParticleTable.GetParticleTable().FindParticle("gamma")
    plist = gamma.GetProcessManager().GetProcessList()
    names = [str(plist[i].GetProcessName()) for i in range(plist.size())]
    simulation_engine.user_hook_log["gamma_processes"] = names


def remove_rindex(simulation_engine):
    """Hook called after the initialization: remove the property table
    (and thus RINDEX) of the water, as if the material had been created
    after the refractive index initialization."""
    import opengate_core as g4

    for mat in g4.G4Material.GetMaterialTable:
        if str(mat.GetName()) == "G4_WATER":
            mat.SetMaterialPropertiesTable(None)


def make_sim(paths, phase_contrast=True, sigma=0.0, energy=None, hook=log_processes):
    sim = gate.Simulation()
    sim.output_dir = paths.output
    sim.number_of_threads = 1
    sim.visu = False
    sim.random_seed = 123456
    sim.world.size = [30 * cm, 30 * cm, 30 * cm]
    sim.world.material = "G4_AIR"
    sim.physics_manager.xray_phase_contrast = phase_contrast
    sim.physics_manager.xray_surface_roughness_sigma = sigma

    box = sim.add_volume("Box", "box")
    box.size = [2 * cm, 2 * cm, 2 * cm]
    box.material = "G4_WATER"

    source = sim.add_source("GenericSource", "src")
    source.particle = "gamma"
    if energy is None:
        source.energy.mono = 20 * keV
    else:
        for k, v in energy.items():
            source.energy[k] = v
    source.position.type = "point"
    source.position.translation = [0, 0, -10 * cm]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.number_of_primaries = 10
    sim.user_hook_after_init = hook
    return sim


def run(sim):
    # Geant4 state persists within a process: each scenario runs in its own
    sim.run(start_new_process=True)
    return sim.user_hook_log


def add_parallel_volume(sim):
    sim.add_parallel_world("world2")
    pbox = sim.add_volume("Box", "pbox")
    pbox.mother = "world2"
    pbox.material = "G4_WATER"
    pbox.size = [1 * cm, 1 * cm, 1 * cm]


def spectrum(max_energy):
    return {
        "type": "histogram",
        "histogram_energy": [10 * keV, 100 * keV, max_energy],
        "histogram_weight": [1.0, 1.0, 1.0],
    }


if __name__ == "__main__":
    paths = utility.get_default_test_paths(__file__, output_folder="test141_engine")

    # child scenarios, launched by the main test (see run_child below)
    if len(sys.argv) > 1 and sys.argv[1] == "--no-rindex":
        sim = make_sim(paths, hook=remove_rindex)
        sim.run(start_new_process=False)
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--negative-sigma":
        make_sim(paths, True, sigma=-0.1).run(start_new_process=False)
        sys.exit(0)

    def run_child(option):
        res = subprocess.run(
            [sys.executable, __file__, option], capture_output=True, text=True
        )
        return res.returncode, res.stdout + res.stderr

    is_ok = True

    # (a) switch on: the process is attached to the gamma
    log = run(make_sim(paths, True))
    ok = "XRayRefraction" in log["gamma_processes"]
    utility.print_test(ok, f"(a) on: gamma processes = {log['gamma_processes']}")
    is_ok &= ok

    # (b) switch off: the process is not there
    log = run(make_sim(paths, False))
    ok = "XRayRefraction" not in log["gamma_processes"]
    utility.print_test(ok, "(b) off: no XRayRefraction process")
    is_ok &= ok

    # (c) negative roughness: exception citing the option. In a child
    # process, because a failing start_new_process hides the message
    code, out = run_child("--negative-sigma")
    ok = code != 0 and "xray_surface_roughness_sigma must be >= 0" in out
    utility.print_test(ok, f"(c) negative sigma: exit code {code}, option cited")
    if not ok:
        print(out[-2000:])
    is_ok &= ok

    # (d) spectrum up to 200 keV: exactly one warning citing 150 keV and 200
    sim = make_sim(paths, True, energy=spectrum(200 * keV))
    run(sim)
    warns = [w for w in sim.warnings if "150 keV" in w and "200" in w]
    ok = len(warns) == 1 and len([w for w in sim.warnings if "150 keV" in w]) == 1
    utility.print_test(ok, f"(d) spectrum to 200 keV: {len(warns)} warning(s)")
    is_ok &= ok

    # (e) spectrum up to 100 keV: no warning
    sim = make_sim(paths, True, energy=spectrum(100 * keV))
    run(sim)
    ok = not [w for w in sim.warnings if "150 keV" in w]
    utility.print_test(ok, "(e) spectrum to 100 keV: no warning")
    is_ok &= ok

    # (f) material without RINDEX: fatal error. The Geant4 fatal aborts the
    # whole process, hence the child process
    code, out = run_child("--no-rindex")
    ok = code != 0 and "XRayRefraction02" in out
    utility.print_test(
        ok, f"(f) no RINDEX: exit code {code}, XRayRefraction02 in output"
    )
    if not ok:
        print(out[-2000:])
    is_ok &= ok

    # (g) parallel world with the contrast on: exactly one warning
    sim = make_sim(paths, True)
    add_parallel_volume(sim)
    run(sim)
    warns = [w for w in sim.warnings if "parallel world" in w]
    ok = len(warns) == 1
    utility.print_test(ok, f"(g) parallel world: {len(warns)} warning(s)")
    is_ok &= ok

    # (h) no parallel world: no such warning
    sim = make_sim(paths, True)
    run(sim)
    ok = not [w for w in sim.warnings if "parallel world" in w]
    utility.print_test(ok, "(h) no parallel world: no warning")
    is_ok &= ok

    utility.test_ok(is_ok)
