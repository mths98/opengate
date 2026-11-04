#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single ray tests of the x-ray refraction process (Snell's law).

The aluminium refractive index decrement is replaced by a constant
delta = 1e-2 (own data folder), so that the deviations are large. The
directions recorded by phase space actors are compared with the vector
form of Snell's law computed here with numpy, using the same refractive
indices as the process (compound_delta, Eq. 21).

The EM physics list option1 has no Rayleigh scattering: a primary photon
that kept the source energy did not interact, so its direction is only
changed by the refraction process.
"""

import pathlib
import shutil

import numpy as np
import uproot
from scipy.spatial.transform import Rotation

import opengate as gate
import opengate.tests.utility as utility
from opengate.utility import g4_units
from opengate.xray_refractive_index import (
    DEFAULT_REFRACTIVE_INDEX_DATA_DIR,
    compound_delta,
    read_reference_densities,
)

keV = g4_units.keV
MeV = g4_units.MeV
mm = g4_units.mm
deg = g4_units.deg

ENERGY = 20 * keV
DELTA_AL = 1e-2
SIGMA = 0.01  # rad, surface roughness of case (e)

# setups placed side by side in the same world, along x
X_SLAB = 0 * mm  # (a) and (d): rotated slab
X_TIR = 60 * mm  # (b): grazing incidence
X_SPHERE = -60 * mm  # (c): sphere
X_SAME = 120 * mm  # (f): same material
TILT = 30 * deg  # (a) rotation of the slab around x
GRAZING = 0.05  # rad, (b) angle between the beam and the surface
IMPACT = 0.5 * mm  # (c) impact parameter
RADIUS = 1 * mm  # (c) sphere radius


def make_data_dir(paths):
    """Copy of the shipped tables where Al has a constant delta = 1e-2."""
    folder = paths.output / "refractive_index"
    if folder.exists():
        shutil.rmtree(folder)
    shutil.copytree(DEFAULT_REFRACTIVE_INDEX_DATA_DIR, folder)
    with open(folder / "Al.txt", "w") as f:
        f.write(f"1.0 {DELTA_AL:.1e} 1e-6\n433.0 {DELTA_AL:.1e} 1e-6\n")
    return folder


def log_composition(simulation_engine):
    """Hook called after the initialization: record the composition of the
    materials, to compute delta as apply_xray_rindex does."""
    import opengate_core as g4

    g_cm3 = g4_units.g / g4_units.cm3
    nist = g4.G4NistManager.Instance()
    composition = {}
    for name in ("G4_Al", "G4_Galactic"):
        mat = nist.FindOrBuildMaterial(name)
        n = mat.GetNumberOfElements()
        symbols = [str(mat.GetElement(i).GetSymbol()) for i in range(n)]
        composition[name] = {
            "symbols": symbols,
            "fractions": [mat.GetElementFraction(i) for i in range(n)],
            "density": mat.GetDensity() / g_cm3,
            "pure_density": [
                nist.FindOrBuildMaterial("G4_" + s).GetDensity() / g_cm3
                for s in symbols
            ],
        }
    simulation_engine.user_hook_log["composition"] = composition


def refractive_index(composition, data_dir):
    """n = 1 - delta at the source energy, as stored in RINDEX."""
    overrides = read_reference_densities(data_dir)
    rho_i = [
        overrides.get(s, rho)
        for s, rho in zip(composition["symbols"], composition["pure_density"])
    ]
    delta = compound_delta(
        composition["symbols"],
        composition["fractions"],
        composition["density"],
        rho_i,
        [ENERGY / keV],
        data_dir,
    )
    return 1.0 - float(delta[0])


def snell(p, normal, n1, n2):
    """Vector form of Snell's law, as in GateXRayRefractionProcess."""
    p = np.asarray(p, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    if np.dot(p, normal) > 0:
        normal = -normal
    r = n1 / n2
    c1 = -np.dot(p, normal)
    k = 1.0 - r * r * (1.0 - c1 * c1)
    if k < 0:
        d = p + 2.0 * c1 * normal
    else:
        d = r * p + (r * c1 - np.sqrt(k)) * normal
    return d / np.linalg.norm(d)


def trace_sphere(origin, p, radius, n_out, n_in):
    """Exit direction of a ray through a sphere centred at the origin."""
    o = np.asarray(origin, dtype=float)
    b = np.dot(o, p)
    t = -b - np.sqrt(b * b - (np.dot(o, o) - radius**2))
    p1 = o + t * p
    d1 = snell(p, p1, n_out, n_in)
    p2 = p1 - 2.0 * np.dot(p1, d1) * d1  # far intersection, |p1| = radius
    return snell(d1, p2, n_in, n_out)


def add_source(sim, name, x, n):
    source = sim.add_source("GenericSource", name)
    source.particle = "gamma"
    source.energy.mono = ENERGY
    source.position.type = "point"
    source.position.translation = [x, 0, -30 * mm]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.number_of_primaries = n
    return source


def add_phsp(sim, label, name, volume):
    phsp = sim.add_actor("PhaseSpaceActor", name)
    phsp.attached_to = volume
    phsp.steps_to_store = "entering"
    phsp.attributes = [
        "PreDirection",
        "PostDirection",
        "KineticEnergy",
        "PostKineticEnergy",
        "TrackID",
    ]
    phsp.output_filename = f"{label}_{name}.root"
    return phsp


def add_slab_with_child(sim, name, x, rotation):
    """Al slab, 1 mm thick, with an Al child in its middle."""
    slab = sim.add_volume("Box", name)
    slab.size = [10 * mm, 10 * mm, 1 * mm]
    slab.translation = [x, 0, 0]
    slab.rotation = rotation
    slab.material = "G4_Al"
    child = sim.add_volume("Box", f"{name}_child")
    child.mother = name
    child.size = [8 * mm, 8 * mm, 0.5 * mm]
    child.material = "G4_Al"
    return slab, child


def add_detector(sim, name, x):
    det = sim.add_volume("Box", name)
    det.size = [20 * mm, 20 * mm, 1 * mm]
    det.translation = [x, 0, 30 * mm]
    det.material = "G4_Galactic"
    return det


def make_sim(paths, label, data_dir, phase_contrast=True, threads=1):
    """Setups (a), (b), (c) and (f), side by side, sigma = 0."""
    sim = gate.Simulation()
    sim.output_dir = paths.output
    sim.number_of_threads = threads
    sim.visu = False
    sim.random_seed = 123456
    sim.world.size = [400 * mm, 100 * mm, 100 * mm]
    sim.world.material = "G4_Galactic"
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option1"
    sim.physics_manager.xray_phase_contrast = phase_contrast
    sim.physics_manager.xray_refractive_index_data_dir = data_dir
    sim.user_hook_after_init = log_composition

    # (a) slab rotated by 30 deg around x, direction in the Al child
    rot = Rotation.from_euler("x", TILT / deg, degrees=True).as_matrix()
    add_slab_with_child(sim, "slab", X_SLAB, rot)
    add_source(sim, "src_slab", X_SLAB, 200)
    add_phsp(sim, label, "phsp_slab", "slab_child")

    # (b) grazing incidence on the large face of a thin box: the box is
    # rotated by 90 deg - GRAZING around x, so that its local -z face
    # passes through (X_TIR, 0, 0) with an angle GRAZING to the beam
    beta = np.pi / 2 - GRAZING
    rot = Rotation.from_euler("x", beta).as_matrix()
    thickness = 1 * mm
    tir = sim.add_volume("Box", "tir")
    tir.size = [10 * mm, 20 * mm, thickness]
    tir.rotation = rot
    tir.translation = list(np.array([X_TIR, 0, 0]) + 0.5 * thickness * rot[:, 2])
    tir.material = "G4_Al"
    add_detector(sim, "det_tir", X_TIR)
    add_source(sim, "src_tir", X_TIR, 50)
    add_phsp(sim, label, "phsp_tir", "det_tir")

    # (c) sphere, ray with impact parameter IMPACT
    sphere = sim.add_volume("Sphere", "sphere")
    sphere.rmax = RADIUS
    sphere.translation = [X_SPHERE, 0, 0]
    sphere.material = "G4_Al"
    add_detector(sim, "det_sphere", X_SPHERE)
    add_source(sim, "src_sphere", X_SPHERE + IMPACT, 300)
    add_phsp(sim, label, "phsp_sphere", "det_sphere")

    # (f) normal incidence on an Al box containing a rotated Al child: the
    # oblique Al -> Al boundaries must not deviate the photon
    same = sim.add_volume("Box", "same")
    same.size = [10 * mm, 10 * mm, 1.5 * mm]
    same.translation = [X_SAME, 0, 0]
    same.material = "G4_Al"
    child = sim.add_volume("Box", "same_child")
    child.mother = "same"
    child.size = [2 * mm, 2 * mm, 0.2 * mm]
    child.rotation = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    child.material = "G4_Al"
    add_source(sim, "src_same", X_SAME, 200)
    add_phsp(sim, label, "phsp_same", "same_child")
    return sim


def make_sim_roughness(paths, label, data_dir, phase_contrast=True):
    """Setup (e): normal incidence on a rough Al slab, direction in the child."""
    sim = gate.Simulation()
    sim.output_dir = paths.output
    sim.number_of_threads = 1
    sim.visu = False
    sim.random_seed = 654321
    sim.world.size = [100 * mm, 100 * mm, 100 * mm]
    sim.world.material = "G4_Galactic"
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option1"
    sim.physics_manager.xray_phase_contrast = phase_contrast
    sim.physics_manager.xray_refractive_index_data_dir = data_dir
    sim.physics_manager.xray_surface_roughness_sigma = SIGMA
    sim.user_hook_after_init = log_composition
    slab, child = add_slab_with_child(sim, "rough", 0, np.eye(3))
    child.size = [8 * mm, 8 * mm, 0.8 * mm]
    add_source(sim, "src_rough", 0, 2400)
    add_phsp(sim, label, "phsp_rough", "rough_child")
    return sim


def run(sim):
    # files of a previous execution would hide an empty output
    for actor in sim.actor_manager.actors.values():
        (pathlib.Path(sim.output_dir) / actor.output_filename).unlink(
            missing_ok=True
        )
    # Geant4 state persists within a process: each run in its own
    sim.run(start_new_process=True)
    return sim.user_hook_log


def read_directions(paths, sim, name, post=False):
    """Pre (or post) step directions of the primaries that did not interact."""
    actor = sim.actor_manager.get_actor(name)
    path = paths.output / actor.output_filename
    if not path.is_file():
        # no photon reached the volume: no file is written
        return np.empty((0, 3))
    with uproot.open(path) as f:
        data = f[name].arrays(library="np")
    e = ENERGY / MeV
    keep = (data["TrackID"] == 1) & (np.abs(data["KineticEnergy"] - e) < 1e-12)
    if post:
        keep &= np.abs(data["PostKineticEnergy"] - e) < 1e-12
    att = "PostDirection" if post else "PreDirection"
    return np.stack([data[f"{att}_{c}"][keep] for c in "XYZ"], axis=1)


def max_deviation(directions, expected):
    if len(directions) == 0:
        return np.inf
    return float(np.max(np.linalg.norm(directions - expected, axis=1)))


def angle(u, v):
    """Angle between unit vectors, accurate for small angles."""
    return np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), np.dot(u, v))


def numpy_roughness(p, n1, n2, sigma, size, rng):
    """Same recipe as the process: Gaussian tilt of the normal with uniform
    azimuth, resampled up to 10 times until c1 > 0, then Snell."""
    normal = np.array([0.0, 0.0, -1.0]) if p[2] > 0 else np.array([0, 0, 1.0])
    u = np.array([1.0, 0.0, 0.0])
    v = np.cross(normal, u)
    out = np.empty((size, 3))
    for i in range(size):
        tilted = normal
        for _ in range(10):
            theta = rng.normal(0.0, sigma)
            phi = 2 * np.pi * rng.uniform()
            t = np.cos(theta) * normal + np.sin(theta) * (
                np.cos(phi) * u + np.sin(phi) * v
            )
            if -np.dot(p, t) > 0:
                tilted = t
                break
        out[i] = snell(p, tilted, n1, n2)
    return out


def report(ok, name, measured, expected, error, tol):
    utility.print_test(
        ok,
        f"{name}: measured {measured} expected {expected} "
        f"error {error:.3e} (tol {tol:.0e})",
    )
    return ok


if __name__ == "__main__":
    paths = utility.get_default_test_paths(
        __file__, output_folder="test141_singleray"
    )
    data_dir = make_data_dir(paths)
    # directions are stored as double in the ROOT file: strict tolerance
    tol = 1e-6
    z = np.array([0.0, 0.0, 1.0])
    is_ok = True

    # ---- process on, sigma = 0: (a), (b), (c), (f)
    sim1 = make_sim(paths, "on", data_dir)
    log = run(sim1)
    comp = log["composition"]
    n_vac = refractive_index(comp["G4_Galactic"], data_dir)
    n_al = refractive_index(comp["G4_Al"], data_dir)
    print(f"n(G4_Galactic) = 1 - {1 - n_vac:.3e}, n(G4_Al) = 1 - {1 - n_al:.6e}")

    # (a) rotated slab: angle with the normal inside Al
    normal = Rotation.from_euler("x", TILT / deg, degrees=True).as_matrix()[:, 2]
    d_slab = read_directions(paths, sim1, "phsp_slab")
    theta_t = np.arcsin(np.sin(TILT) * n_vac / n_al)
    measured = angle(d_slab, normal)
    rel = np.max(np.abs(measured - theta_t)) / theta_t if len(d_slab) else np.inf
    expected_slab = snell(z, normal, n_vac, n_al)
    dev = max_deviation(d_slab, expected_slab)
    ok = len(d_slab) > 50 and rel < tol and dev < tol
    is_ok &= report(
        ok,
        f"(a) rotated slab, {len(d_slab)} photons, angle to normal",
        f"{np.mean(measured):.12f}",
        f"{theta_t:.12f}",
        max(rel, dev),
        tol,
    )

    # (b) total external reflection: specular direction
    d_tir = read_directions(paths, sim1, "phsp_tir")
    beta = np.pi / 2 - GRAZING
    normal_tir = Rotation.from_euler("x", beta).as_matrix()[:, 2]
    critical = np.arccos(n_al / n_vac)
    expected_tir = z - 2 * np.dot(z, normal_tir) * normal_tir
    dev = max_deviation(d_tir, expected_tir)
    ok = GRAZING < critical and len(d_tir) == 50 and dev < tol
    is_ok &= report(
        ok,
        f"(b) TIR, grazing {GRAZING} < critical {critical:.4f} rad, "
        f"{len(d_tir)} photons",
        d_tir[0] if len(d_tir) else None,
        expected_tir,
        dev,
        tol,
    )

    # (c) sphere: exit direction after two refractions
    d_sph = read_directions(paths, sim1, "phsp_sphere")
    expected_sph = trace_sphere([IMPACT, 0, -30 * mm], z, RADIUS, n_vac, n_al)
    dev = max_deviation(d_sph, expected_sph)
    ok = len(d_sph) > 20 and dev < tol
    is_ok &= report(
        ok,
        f"(c) sphere b = {IMPACT} mm, {len(d_sph)} photons",
        d_sph[0] if len(d_sph) else None,
        expected_sph,
        dev,
        tol,
    )

    # (f) same material: Al -> Al oblique boundaries, no deviation. Snell
    # with n1 = n2 is the identity up to round-off, so the direction must be
    # exactly untouched: the process skips the boundary
    d_same = read_directions(paths, sim1, "phsp_same")
    d_same_post = read_directions(paths, sim1, "phsp_same", post=True)
    dev = max(max_deviation(d_same, z), max_deviation(d_same_post, z))
    ok = len(d_same) > 50 and len(d_same_post) > 50 and dev == 0.0
    is_ok &= report(
        ok,
        f"(f) same material, {len(d_same)} entering / "
        f"{len(d_same_post)} exiting the child",
        d_same[0] if len(d_same) else None,
        z,
        dev,
        0.0,
    )

    # ---- (d) process off: direction unchanged
    sim_off = make_sim(paths, "off", data_dir, phase_contrast=False)
    run(sim_off)
    d_off = read_directions(paths, sim_off, "phsp_slab")
    dev = max_deviation(d_off, z)
    ok = len(d_off) > 50 and dev < 1e-12
    first = d_off[0] if len(d_off) else None
    is_ok &= report(ok, f"(d) off, {len(d_off)} photons", first, z, dev, 1e-12)

    # ---- (e) roughness: spread of the refracted direction
    sim_r = make_sim_roughness(paths, "rough", data_dir)
    run(sim_r)
    d_rough = read_directions(paths, sim_r, "phsp_rough")
    rng = np.random.default_rng(42)
    d_ref = numpy_roughness(z, n_vac, n_al, SIGMA, 20000, rng)
    std_sim = np.std(angle(d_rough, z))
    std_ref = np.std(angle(d_ref, z))
    rel = abs(std_sim - std_ref) / std_ref
    ok = len(d_rough) >= 2000 and rel < 0.1
    is_ok &= report(
        ok,
        f"(e) roughness sigma = {SIGMA} rad, {len(d_rough)} photons, "
        "std of the exit angle",
        f"{std_sim:.4e}",
        f"{std_ref:.4e}",
        rel,
        0.1,
    )

    # ---- (g) 4 threads: (a) and (c) give the same direction as 1 thread
    sim_mt = make_sim(paths, "mt", data_dir, threads=4)
    run(sim_mt)
    for name, d1 in (("phsp_slab", d_slab), ("phsp_sphere", d_sph)):
        d4 = read_directions(paths, sim_mt, name)
        dev = max_deviation(d4, d1[0]) if len(d1) else np.inf
        dev = max(dev, max_deviation(d1, d1[0]) if len(d1) else np.inf)
        ok = len(d4) > 20 and dev < 1e-12
        is_ok &= report(
            ok,
            f"(g) {name}, 1 vs 4 threads, {len(d1)} / {len(d4)} photons",
            d4[0] if len(d4) else None,
            d1[0] if len(d1) else None,
            dev,
            1e-12,
        )

    utility.test_ok(is_ok)
