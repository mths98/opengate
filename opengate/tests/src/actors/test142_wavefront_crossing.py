#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""First crossing, kill_after_plane, MT and robustness of the WavefrontActor.

Simulation A (option4, 4 threads, two runs, sources by activity):
  (d)       lead behind the plane: every gamma track counts once, so N is the
            number of distinct (RunID, EventID, TrackID) entering a thin volume at the
            plane (PhaseSpaceActor), and n_recrossings > 0; without lead (control)
            n_recrossings = 0 and N = emitted
  focus 1   the phase is accumulated step by step with the energy of the step:
            a thin Al scatterer (Compton changes the energy) followed by a PMMA
            plate, the photons that interact in the plate are killed
  focus 2   the two runs accumulate in the same grids
Simulation B (vacuum, no interactions, point source so that the pixels are
reproducible), run with 1 thread and with 4 threads + kill_after_plane:
  (g)       the same grids and counters with 1 and 4 threads; the disc source
            only gives the same N and sums, and every thread gets events
  (e)       kill_after_plane: the photons stop at the plane, the grids do not
            change
  focus 4   the actor ignores electrons that do cross the plane (inside its grid)
  focus 5   two actors with different planes keep their own state and files
"""

import json

import numpy as np
import uproot

import opengate as gate
import opengate.tests.utility as utility
from opengate.sources.base import SourceBase as Source
from opengate.utility import g4_units
from test142_wavefront_basic import (
    C_LIGHT,
    H_PLANCK,
    g4_linear_value,
    log_materials,
    rindex_table,
)

keV = g4_units.keV
mm = g4_units.mm
um = g4_units.um
sec = g4_units.s
Bq = g4_units.Bq

GRID = [20, 20]
SPACING = [0.1 * mm, 0.1 * mm]
PLANE_Z = 100 * mm


def check(ok, text):
    utility.print_test(ok, text)
    return ok


def make_sim(paths, name, physics="G4EmStandardPhysics_option1", threads=1):
    sim = gate.Simulation()
    sim.output_dir = paths.output / name
    sim.number_of_threads = threads
    sim.visu = False
    sim.random_seed = 123456
    sim.world.size = [400 * mm, 100 * mm, 300 * mm]
    sim.world.material = "G4_Galactic"
    sim.physics_manager.physics_list_name = physics
    sim.physics_manager.xray_phase_contrast = True
    return sim


def no_interactions(sim):
    sim.g4_commands_after_init.append("/process/inactivate GammaGeneralProc")


def add_source(sim, name, x, n, particle="gamma", energy=20 * keV):
    """Beam along +z from (x, 0, 0); n is the number of primaries (a list with
    one number per run if there are several runs)."""
    source = sim.add_source("GenericSource", name)
    source.particle = particle
    source.energy.mono = energy
    source.position.type = "disc"
    source.position.radius = 0.8 * mm
    source.position.translation = [x, 0, 0]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    if n is not None:
        source.number_of_primaries = n
    return source


def make_point(source, x, y, z=0):
    source.position.type = "point"
    source.position.translation = [x, y, z]


def add_box(sim, name, x, z, size_xy, thickness, material):
    box = sim.add_volume("Box", name)
    box.size = [size_xy, size_xy, thickness]
    box.translation = [x, 0, z]
    box.material = material
    return box


def add_wavefront(sim, name, x=0, plane_z=PLANE_Z, kill=False, grid=GRID):
    wf = sim.add_actor("WavefrontActor", name)
    wf.attached_to = "world"
    wf.plane_z = plane_z
    wf.plane_center = [x, 0]
    wf.size = grid
    wf.spacing = SPACING
    wf.kill_after_plane = kill
    wf.output_filename = f"{name}.mhd"
    return wf


def add_fluence(sim, name, x, z):
    """FluenceActor in a thin vacuum box: tells whether the particles go on."""
    thin = add_box(sim, f"thin_{name}", x, z, 10 * mm, 1 * mm, "G4_Galactic")
    flu = sim.add_actor("FluenceActor", name)
    flu.attached_to = thin.name
    flu.size = [1, 1, 1]
    flu.spacing = [10 * mm, 10 * mm, 1 * mm]
    flu.output_filename = f"{name}.mhd"


def after_run(simulation_engine):
    """Hook after the run: record the composition of the materials and the
    totals of the FluenceActors. An ITK image cannot be pickled back to the
    process, so the fluence outputs are released here."""
    import itk

    log = simulation_engine.user_hook_log
    if simulation_engine.simulation.physics_manager.xray_phase_contrast:
        log_materials(simulation_engine)
    log["fluence"] = {}
    for name, actor in simulation_engine.simulation.actor_manager.actors.items():
        if not name.startswith("flu"):
            continue
        image = itk.array_from_image(actor.counts.image)
        log["fluence"][name] = float(image.sum())
        for attr in vars(actor):
            if attr.startswith("list_of_"):
                setattr(actor, attr, [])
        for out in actor.user_output.values():
            out.data_per_run = {}
            out.merged_data = None


def grids(wf):
    return wf.real.copy(), wf.imag.copy(), wf.energy.copy()


def phase_of(table, energy_keV, thickness_mm):
    """-2 pi delta(E) T / lambda with delta from an (energies, n) table."""
    energy = energy_keV * keV
    delta = 1.0 - g4_linear_value(table[0], table[1], energy)
    lam = H_PLANCK * C_LIGHT / energy
    return -2.0 * np.pi * delta * (thickness_mm * mm) / lam


def distinct_gamma_tracks(path):
    """Number of distinct (RunID, EventID, TrackID) of the gammas."""
    with uproot.open(path) as f:
        tree = f[f.keys()[0]]
        names = ["RunID", "EventID", "TrackID", "ParticleName"]
        data = tree.arrays(names, library="np")
    gamma = data["ParticleName"] == "gamma"
    keys = (data[k][gamma].tolist() for k in ("RunID", "EventID", "TrackID"))
    pairs = set(zip(*keys))
    return len(pairs), int(gamma.sum())


# ---------------------------------------------------------------- simulation A
# beams along x: lead (-120), vacuum (-60), scatterer (0), control (+120)
AL_T = 10 * um
PRE_T = 1 * mm  # plate before the scatterer, normal incidence
Z_PRE = -3 * mm
PMMA_T = 1 * mm  # plate after the scatterer, seen at the polar angle
Z_PMMA = 5.5 * mm
F1_PLANE = 20 * mm
F1_ENERGY = 60
F1_PIXELS = 600
F1_SPACING = 0.1 * mm
F1_ORIGIN = -0.5 * mm  # corner of the grid, so that the beam is in pixel (5, 5)
F1_BEAM = 0.05 * mm


def make_sim_a(paths):
    sim = make_sim(paths, "A", "G4EmStandardPhysics_option4", threads=4)
    sim.run_timing_intervals = [[0, 1 * sec], [1 * sec, 2 * sec]]
    sim.user_hook_after_run = after_run

    # lead beam: a thin vacuum box around the plane, the lead right behind it
    add_source(sim, "src_pb", -120 * mm, n=[5000, 5000])
    z_pb = PLANE_Z + 0.05 * mm + 2.5 * mm
    add_box(sim, "pb", -120 * mm, z_pb, 20 * mm, 5 * mm, "G4_Pb")
    add_wavefront(sim, "wf_pb", -120 * mm, grid=[200, 200])
    box = add_box(sim, "box_pb", -120 * mm, PLANE_Z, 20 * mm, 0.1 * mm, "G4_Galactic")
    phsp = sim.add_actor("PhaseSpaceActor", "phsp_pb")
    phsp.attached_to = box.name
    phsp.attributes = ["RunID", "EventID", "TrackID", "ParticleName"]
    phsp.steps_to_store = "entering"
    phsp.output_filename = "phsp_pb.root"

    # control: the same without lead
    add_source(sim, "src_ctrl", 120 * mm, n=[1500, 1500])
    # (behind the lead of the other beam: its backscattered photons do not get
    # there, and n_recrossings counts all the photons that cross the plane)
    add_wavefront(sim, "wf_ctrl", 120 * mm, plane_z=140 * mm, grid=[200, 200])

    # focus 2: vacuum beam, two runs
    add_source(sim, "src_vac", -60 * mm, n=[2000, 2000])
    add_wavefront(sim, "wf_vac", -60 * mm)

    # focus 1: Al scatterer at z = 0, PMMA plate, vacuum, plane
    src = add_source(sim, "src_f1", 0, n=[200000, 200000], energy=F1_ENERGY * keV)
    make_point(src, F1_BEAM, F1_BEAM, -6 * mm)
    add_box(sim, "al", 0, 0, 20 * mm, AL_T, "G4_Al")
    # photons that interact in a plate are killed, so that the phase of
    # the photons that arrive is only the one of the steps without interaction
    for tag, z, thick in (("pre", Z_PRE, PRE_T), ("post", Z_PMMA, PMMA_T)):
        plate = add_box(sim, f"pmma_{tag}", 0, z, 80 * mm, thick, "G4_PLEXIGLASS")
        kill = sim.add_actor("KillAccordingProcessesActor", f"kill_{tag}")
        kill.attached_to = plate.name
        kill.processes_to_kill = ["all"]
        kill.is_rayleigh_an_interaction = True
    half = F1_PIXELS * F1_SPACING / 2
    wf = add_wavefront(
        sim,
        "wf_f1",
        0,
        plane_z=F1_PLANE,
        grid=[F1_PIXELS, F1_PIXELS],
    )
    wf.plane_center = [F1_ORIGIN + half, F1_ORIGIN + half]
    wf.spacing = [F1_SPACING, F1_SPACING]
    return sim


def wrap(x):
    return float(np.angle(np.exp(1j * x)))


def check_phase_per_step(wf, comp):
    """Compton photons that are alone in their pixel. The expected phase is
    P1(E0) + (a1 + a2) / 2 + P2(E'), with P1 the plate before the scatterer
    (normal incidence, the photon still has E0), P2 the plate after it (path
    T / cos(theta), energy E'), and a1 = phase of the scatterer at E0 and
    a2 = phase of the scatterer after the scattering (E', 1 / cos(theta)); the
    scattering point is somewhere inside the scatterer, so the phase of the
    scatterer is between a1 and a2. Two wrong models must fail in every pixel:
    (a) E0 at all the steps, (b) E' at all the steps."""
    pmma = rindex_table(comp["G4_PLEXIGLASS"])
    al = rindex_table(comp["G4_Al"])
    psi = wf.real + 1j * wf.imag
    energy = wf.energy
    e0 = F1_ENERGY
    single = (energy > 0) & (np.abs(np.abs(psi) ** 2 - energy) < 1e-6 * energy)
    single &= energy < e0 - 0.5  # Compton
    d = F1_SPACING / mm
    x0, beam, dz = F1_ORIGIN / mm, F1_BEAM / mm, F1_PLANE / mm
    a1 = phase_of(al, e0, AL_T / mm)

    def models(e, x, y):
        cos_t = dz / np.sqrt((x - beam) ** 2 + (y - beam) ** 2 + dz**2)
        a2 = phase_of(al, e, AL_T / mm / cos_t)
        scat = (a1 + a2) / 2
        right = phase_of(pmma, e0, PRE_T / mm) + scat + phase_of(
            pmma, e, PMMA_T / mm / cos_t
        )
        wrong_a = phase_of(pmma, e0, PRE_T / mm) + scat + phase_of(
            pmma, e0, PMMA_T / mm / cos_t
        )
        wrong_b = phase_of(pmma, e, PRE_T / mm) + scat + phase_of(
            pmma, e, PMMA_T / mm / cos_t
        )
        return right, wrong_a, wrong_b, abs(a1 - a2) / 2, cos_t

    rows, n_single, n_low = [], 0, 0
    for j, i in zip(*np.nonzero(single)):
        e = float(energy[j, i])
        n_single += 1
        x, y = x0 + (i + 0.5) * d, x0 + (j + 0.5) * d
        right, wrong_a, wrong_b, half_a, cos_t = models(e, x, y)
        corners = [
            models(e, x + sx * d / 2, y + sy * d / 2)[0]
            for sx in (-1, 1)
            for sy in (-1, 1)
        ]
        tol = half_a + max(abs(c - right) for c in corners) + 0.02
        # selection from the expected values: both wrong models must be
        # more than 2 tol away from the right one (modulo 2 pi)
        if min(abs(wrap(wrong_a - right)), abs(wrap(wrong_b - right))) <= 2 * tol:
            n_low += 1
            continue
        res = [abs(wrap(np.angle(psi[j, i]) - m)) for m in (right, wrong_a, wrong_b)]
        rows.append((np.degrees(np.arccos(cos_t)), e, *res, tol))
    print("   theta[deg]  E'[keV]  res_right  res_E0  res_E'    tol   [rad]")
    for r in sorted(rows):
        print("   " + "  ".join(f"{v:9.3f}" for v in r))
    ok = len(rows) >= 10
    ok &= all(r[2] <= r[5] and r[3] > r[5] and r[4] > r[5] for r in rows)
    return check(
        ok,
        f"(focus 1) {len(rows)} Compton pixels selected ({n_single} Compton "
        f"single-photon pixels, {n_low} discarded because a wrong model would "
        f"be within 2 tol); right model passes in all, E0-everywhere and "
        f"E'-everywhere fail in all",
    )


def case_a(paths):
    sim = make_sim_a(paths)
    sim.run(start_new_process=True)
    get = sim.actor_manager.get_actor
    log = sim.user_hook_log
    emitted = {"src_pb": 10000, "src_ctrl": 3000, "src_vac": 4000}

    # (d) lead: every gamma track counts once
    wf = get("wf_pb")
    c = wf.counters
    tracks, hits = distinct_gamma_tracks(sim.output_dir / "phsp_pb.root")
    ok = c["n_recrossings"] > 0 and c["n_photons_counted"] == tracks
    ok &= c["n_photons_counted"] >= emitted["src_pb"]
    ok_d = check(
        ok,
        f"(d) lead: N = {c['n_photons_counted']}, distinct gamma tracks entering "
        f"the plane volume = {tracks} ({hits} entries), emitted = "
        f"{emitted['src_pb']}, n_recrossings = {c['n_recrossings']}",
    )
    c = get("wf_ctrl").counters
    ok_c = check(
        c["n_recrossings"] == 0 and c["n_photons_counted"] == emitted["src_ctrl"],
        f"(d) control without lead: N = {c['n_photons_counted']}, emitted = "
        f"{emitted['src_ctrl']}, n_recrossings = {c['n_recrossings']}",
    )

    # focus 2: two runs
    wf = get("wf_vac")
    n = wf.counters["n_photons_counted"]
    ok = n == emitted["src_vac"] > 0
    ok &= abs(float(wf.energy.sum()) - 20.0 * n) < 1e-6
    ok_2 = check(
        ok,
        f"(focus 2) two runs: N = {n}, emitted = {emitted['src_vac']}, "
        f"sum E = {float(wf.energy.sum()):.1f}",
    )

    ok_1 = check_phase_per_step(get("wf_f1"), log["materials"])
    return ok_d and ok_c and ok_2 and ok_1


# ---------------------------------------------------------------- simulation B
N_POINT = 4000
N_DISC = 1000
N_ELECTRONS = 500
X_DISC = -30 * mm
X_ELECTRONS = 30 * mm
Z_PMMA_B = 75 * mm
T_PMMA_B = 1 * mm


def run_vacuum(paths, name, threads, kill):
    """Point source (two actors, planes at 50 and 100 mm, a PMMA plate between
    them), disc source, electron source, FluenceActors behind the planes."""
    sim = make_sim(paths, name, threads=threads)
    no_interactions(sim)
    sim.user_hook_after_run = after_run
    src = add_source(sim, "src_point", 0, n=N_POINT)
    make_point(src, 0.55 * mm, -0.35 * mm)
    add_source(sim, "src_disc", X_DISC, n=N_DISC)
    add_source(sim, "src_e", X_ELECTRONS, n=N_ELECTRONS, particle="e-")
    add_box(sim, "pmma", 0, Z_PMMA_B, 20 * mm, T_PMMA_B, "G4_PLEXIGLASS")
    add_wavefront(sim, "wf100", plane_z=PLANE_Z, kill=kill)
    add_wavefront(sim, "wf50", plane_z=50 * mm)
    add_wavefront(sim, "wf_disc", X_DISC, plane_z=60 * mm)
    add_wavefront(sim, "wf_e", X_ELECTRONS, plane_z=PLANE_Z)
    add_fluence(sim, "flu", 0, 110 * mm)
    add_fluence(sim, "flu_e", X_ELECTRONS, 110 * mm)
    sim.run(start_new_process=True)
    get = sim.actor_manager.get_actor
    result = {
        n: (grids(get(n)), dict(get(n).counters))
        for n in ("wf100", "wf50", "wf_disc", "wf_e")
    }
    return result, sim.user_hook_log, sim


def case_b(paths):
    r1, log1, sim1 = run_vacuum(paths, "B1", threads=1, kill=False)
    r4, log4, _ = run_vacuum(paths, "B4", threads=4, kill=True)

    # (g) 1 vs 4 threads: point source gives identical grids and counters
    err = 0.0
    ok = True
    for name in ("wf100", "wf50"):
        (g1, c1), (g4, c4) = r1[name], r4[name]
        ok &= c1 == c4 and c1["n_photons_counted"] == N_POINT
        for a, b in zip(g1, g4):
            scale = max(float(np.max(np.abs(a))), 1e-300)
            err = max(err, float(np.max(np.abs(a - b))) / scale)
            sa, sb = float(a.sum()), float(b.sum())
            err = max(err, abs(sa - sb) / max(abs(sa), 1e-300))
    ok_g = check(
        ok and err < 1e-12,
        f"(g) point source, 1 vs 4 threads: N = {N_POINT}, counters equal, "
        f"relative error {err:.3e}",
    )
    # disc source: only N and the sums, and every thread got events
    (g1, c1), (g4, c4) = r1["wf_disc"], r4["wf_disc"]
    err = max(
        abs(float(a.sum()) - float(b.sum())) / max(abs(float(a.sum())), 1e-300)
        for a, b in zip(g1, g4)
    )
    err = max(err, abs(c1["sum_energy_keV"] - c4["sum_energy_keV"]) / 20.0 / N_DISC)
    # Source._scale_counts_for_thread is what splits the primaries between
    # the threads (the counters of the C++ source managers are not readable
    # after the run), so this checks that more than one thread has events
    per_thread = [int(Source._scale_counts_for_thread(N_DISC, i, 4)) for i in range(4)]
    ok = c1["n_photons_counted"] == c4["n_photons_counted"] == N_DISC
    ok &= err < 1e-12 and sum(per_thread) == N_DISC
    ok &= sum(1 for e in per_thread if e > 0) > 1
    ok_gd = check(
        ok,
        f"(g) disc source: N = {c4['n_photons_counted']} in both, relative error "
        f"of the sums {err:.3e}, events per thread (split rule) {per_thread}",
    )

    # (e) kill_after_plane (4 threads, True) vs no kill (1 thread, False)
    flu1, flu4 = log1["fluence"]["flu"], log4["fluence"]["flu"]
    ok = flu4 == 0 and flu1 == N_POINT
    ok_e = check(
        ok,
        f"(e) kill_after_plane: fluence behind the plane True = {flu4}, "
        f"False = {flu1}, grids identical to the run without kill (see g)",
    )

    # focus 4: electrons cross the plane inside the grid and are ignored
    c = r1["wf_e"][1]
    ok = log1["fluence"]["flu_e"] == N_ELECTRONS
    ok &= c["n_photons_counted"] == 0 and c["n_recrossings"] == 0
    ok &= c["n_outside_grid"] == N_POINT + N_DISC and c["sum_energy_keV"] == 0
    ok_4 = check(
        ok,
        f"(focus 4) {N_ELECTRONS} electrons of 20 keV crossed the plane (fluence "
        f"behind it {log1['fluence']['flu_e']}): counters {c}",
    )

    # focus 5: own state and own files, different planes and phases
    comp = log1["materials"]
    pmma = rindex_table(comp["G4_PLEXIGLASS"])
    expected = {"wf50": 0.0, "wf100": phase_of(pmma, 20, T_PMMA_B / mm)}
    ok = True
    info = []
    planes = []
    for name in ("wf50", "wf100"):
        (re, im, en), c = r1[name]
        psi = (re + 1j * im)[en > 0]
        res = float(np.max(np.abs(np.angle(psi * np.exp(-1j * expected[name])))))
        ok &= res < 1e-9 and c["n_photons_counted"] == N_POINT
        suffixes = ("_real.mhd", "_imag.mhd", "_energy.mhd", ".json")
        files = [f"{name}{s}" for s in suffixes]
        ok &= all((sim1.output_dir / f).is_file() for f in files)
        with open(sim1.output_dir / f"{name}.json") as f:
            meta = json.load(f)
        planes.append(meta["plane_z_mm"])
        info.append(f"{name}: phase residual {res:.1e}")
    ok &= planes == [50.0, 100.0]
    ok_5 = check(ok, f"(focus 5) planes in JSON {planes}, {info}, own files written")
    return ok_g and ok_gd and ok_e and ok_4 and ok_5


if __name__ == "__main__":
    paths = utility.get_default_test_paths(__file__, output_folder="test142_crossing")
    is_ok = True
    for case in (case_a, case_b):
        is_ok &= case(paths)
    utility.test_ok(is_ok)
