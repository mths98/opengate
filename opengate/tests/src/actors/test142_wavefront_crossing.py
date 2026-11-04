#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""First crossing, kill_after_plane, MT and robustness of the WavefrontActor.

(d) backscattering by lead behind the plane: re-crossings are counted
(e) kill_after_plane: photons stop at the plane, the grids do not change
(g) 1 and 4 threads give the same grids and the same N
Focus 1: Compton before the plane lowers the effective energy
Focus 2: two runs accumulate in the same grids
Focus 4: secondary electrons are ignored
Focus 5: two actors with different planes keep their own state
"""

import numpy as np

import opengate as gate
import opengate.tests.utility as utility
from opengate.utility import g4_units

keV = g4_units.keV
mm = g4_units.mm
sec = g4_units.s
Bq = g4_units.Bq

N_PHOTONS = 2000
GRID = [20, 20]
SPACING = [0.1 * mm, 0.1 * mm]


def make_sim(paths, name, physics="G4EmStandardPhysics_option1", threads=1):
    sim = gate.Simulation()
    sim.output_dir = paths.output / name
    sim.number_of_threads = threads
    sim.visu = False
    sim.random_seed = 123456
    sim.world.size = [100 * mm, 100 * mm, 300 * mm]
    sim.world.material = "G4_Galactic"
    sim.physics_manager.physics_list_name = physics
    sim.physics_manager.xray_phase_contrast = True
    return sim


def no_interactions(sim):
    sim.g4_commands_after_init.append("/process/inactivate GammaGeneralProc")


def add_source(
    sim, n=N_PHOTONS, particle="gamma", energy=20 * keV, point=False, name="src"
):
    source = sim.add_source("GenericSource", name)
    source.particle = particle
    source.energy.mono = energy
    if point:
        source.position.type = "point"
        source.position.translation = [0.55 * mm, -0.35 * mm, 0]
    else:
        source.position.type = "disc"
        source.position.radius = 0.8 * mm
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.number_of_primaries = n
    return source


def add_slab(sim, name, z, thickness, material):
    slab = sim.add_volume("Box", name)
    slab.size = [20 * mm, 20 * mm, thickness]
    slab.translation = [0, 0, z]
    slab.material = material
    return slab


def add_wavefront(sim, name="wf", plane_z=100 * mm, kill=False):
    wf = sim.add_actor("WavefrontActor", name)
    wf.attached_to = "world"
    wf.plane_z = plane_z
    wf.size = GRID
    wf.spacing = SPACING
    wf.kill_after_plane = kill
    wf.output_filename = f"{name}.mhd"
    return wf


def add_stats(sim):
    return sim.add_actor("SimulationStatisticsActor", "stats")


def grids(wf):
    return wf.real.copy(), wf.imag.copy(), wf.energy.copy()


def check(ok, text):
    utility.print_test(ok, text)
    return ok


def case_scattering(paths):
    """Interactions on (option4), two beams in one simulation:
    (d) lead block 5 mm right behind the plane (x = -120 mm): backscattered
        photons cross the plane again, N <= emitted;
    focus 1: water slab 5 mm before the plane (x = +120 mm): the Compton
        photons reach the plane with less than 20 keV."""
    sim = make_sim(paths, "scattering", "G4EmStandardPhysics_option4")
    # beams far apart: no photon scattered by one reaches the other plane
    sim.world.size = [300 * mm, 100 * mm, 300 * mm]
    for tag, x, z_slab, thick, mat, z_plane in (
        ("pb", -120 * mm, 102.5 * mm, 5 * mm, "G4_Pb", 100 * mm),
        ("water", 120 * mm, 5 * mm, 5 * mm, "G4_WATER", 10 * mm),
    ):
        source = add_source(sim, n=5000, name=f"src_{tag}")
        source.position.translation = [x, 0, 0]
        slab = add_slab(sim, f"slab_{tag}", z_slab, thick, mat)
        slab.translation = [x, 0, z_slab]
        wf = add_wavefront(sim, f"wf_{tag}", plane_z=z_plane)
        wf.plane_center = [x, 0]
        wf.size = [200, 200]
        wf.spacing = [0.1 * mm, 0.1 * mm]
    sim.run(start_new_process=True)
    get = sim.actor_manager.get_actor

    c = get("wf_pb").counters
    n = c["n_photons_counted"]
    ok_d = check(
        c["n_recrossings"] > 0 and 0 < n <= 5000,
        f"(d) backscatter by Pb: N = {n} (emitted 5000), "
        f"n_recrossings = {c['n_recrossings']}",
    )

    wf = get("wf_water")
    c = wf.counters
    eff = c["sum_energy_keV"] / c["n_photons_counted"]
    # a pixel hit by a single 20 keV photon has sum E = 20: less than that
    # means that some photon arrived with less energy
    n_lt = int(np.sum((wf.energy > 0) & (wf.energy < 20.0 - 1e-9)))
    ok_1 = check(
        eff < 20 and n_lt > 0,
        f"(focus 1) Compton before the plane: effective energy {eff:.4f} keV "
        f"(< 20), pixels with sum E < 20 keV: {n_lt}, N = {c['n_photons_counted']}",
    )
    return ok_d and ok_1


def log_fluence(simulation_engine):
    """Hook after the run: total of the FluenceActor behind the plane."""
    import itk

    actor = simulation_engine.simulation.actor_manager.get_actor("flu")
    image = itk.array_from_image(actor.counts.image)
    simulation_engine.user_hook_log["fluence"] = float(image.sum())
    # an ITK image cannot be pickled back to the parent process
    for name in vars(actor):
        if name.startswith("list_of_"):
            setattr(actor, name, [])
    for out in actor.user_output.values():
        out.data_per_run = {}
        out.merged_data = None


def run_vacuum(paths, name, threads=1, kill=False):
    """Point source through PMMA, in a world without interactions, with two
    actors (focus 5), an electron source (focus 4) and a FluenceActor in a
    thin volume behind the plane."""
    sim = make_sim(paths, name, threads=threads)
    no_interactions(sim)
    add_source(sim, n=4000, point=True)
    add_source(sim, n=500, particle="e-", name="src_electrons")
    add_slab(sim, "pmma", 20 * mm, 1 * mm, "G4_PLEXIGLASS")
    add_wavefront(sim, "wf", plane_z=100 * mm, kill=kill)
    add_wavefront(sim, "wf50", plane_z=50 * mm)
    add_slab(sim, "thin", 110 * mm, 1 * mm, "G4_Galactic")
    flu = sim.add_actor("FluenceActor", "flu")
    flu.attached_to = "thin"
    flu.size = [1, 1, 1]
    flu.spacing = [20 * mm, 20 * mm, 1 * mm]
    flu.output_filename = "flu.mhd"
    sim.user_hook_after_run = log_fluence
    sim.run(start_new_process=True)
    get = sim.actor_manager.get_actor
    result = {n: (grids(get(n)), get(n).counters) for n in ("wf", "wf50")}
    return result, sim.user_hook_log["fluence"], sim


def case_vacuum(paths):
    r1, flu1, sim1 = run_vacuum(paths, "vacuum_1")
    r4, _, _ = run_vacuum(paths, "vacuum_4", threads=4)
    rk, flu_k, _ = run_vacuum(paths, "vacuum_kill", kill=True)

    # (g) 1 vs 4 threads
    ok = True
    err = 0.0
    for name in r1:
        (g1, c1), (g4, c4) = r1[name], r4[name]
        ok &= c1["n_photons_counted"] == c4["n_photons_counted"] == 4000
        ok &= c1["n_outside_grid"] == c4["n_outside_grid"]
        for a, b in zip(g1, g4):
            scale = max(float(np.max(np.abs(a))), 1e-300)
            err = max(err, float(np.max(np.abs(a - b))) / scale)
            sa, sb = float(a.sum()), float(b.sum())
            err = max(err, abs(sa - sb) / max(abs(sa), 1e-300))
    ok &= err < 1e-12
    n1 = {n: r1[n][1]["n_photons_counted"] for n in r1}
    ok_g = check(ok, f"(g) 1 vs 4 threads: N = {n1}, relative error {err:.3e}")

    # focus 4 and 5: electrons ignored, own state and files per actor
    ok_f = all(
        (sim1.output_dir / f"{n}{suffix}").is_file()
        for n in ("wf", "wf50")
        for suffix in ("_real.mhd", "_imag.mhd", "_energy.mhd", ".json")
    )
    ok_f &= n1["wf"] == n1["wf50"] == 4000
    ok_f &= r1["wf"][1]["n_outside_grid"] == 0
    ok_f = check(
        ok_f,
        f"(focus 4, 5) electrons ignored, two planes: N = {n1} "
        "(photons emitted: 4000), own files written",
    )

    # (e) kill_after_plane: same source and seed, in vacuum
    gk, ck = rk["wf"]
    g1_, c1_ = r1["wf"]
    ok = flu_k == 0 and flu1 > 0
    ok &= all(np.array_equal(a, b) for a, b in zip(gk, g1_))
    ok &= ck == c1_
    ok_e = check(
        ok,
        f"(e) kill_after_plane: fluence behind plane True = {flu_k}, "
        f"False = {flu1}, grids and counters identical, N = {ck['n_photons_counted']}",
    )
    return ok_g and ok_f and ok_e


def case_focus2(paths):
    sim = make_sim(paths, "f2")
    no_interactions(sim)
    src = add_source(sim, n=0)
    src.activity = 1000 * Bq
    sim.run_timing_intervals = [[0, 1 * sec], [1 * sec, 2 * sec]]
    add_wavefront(sim)
    stats = add_stats(sim)
    sim.run(start_new_process=True)
    wf = sim.actor_manager.get_actor("wf")
    emitted = sim.actor_manager.get_actor("stats").counts.events
    n = wf.counters["n_photons_counted"]
    ok = n == emitted and emitted > 0
    ok &= abs(float(wf.energy.sum()) - 20.0 * n) < 1e-6
    ok &= abs(wf.counters["sum_energy_keV"] - 20.0 * n) < 1e-6
    return check(
        ok, f"(focus 2) two runs: N = {n}, emitted = {emitted}, sum E = "
        f"{float(wf.energy.sum()):.1f}"
    )


if __name__ == "__main__":
    paths = utility.get_default_test_paths(__file__, output_folder="test142_crossing")
    is_ok = True
    for case in (case_scattering, case_vacuum, case_focus2):
        is_ok &= case(paths)
    utility.test_ok(is_ok)
