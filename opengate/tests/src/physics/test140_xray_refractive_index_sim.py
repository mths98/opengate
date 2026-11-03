#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil

import numpy as np
import itk

import opengate as gate
import opengate.tests.utility as utility
from opengate.utility import g4_units

keV = g4_units.keV
cm = g4_units.cm
mm = g4_units.mm


def log_materials(simulation_engine):
    """Hook called after the initialization: record the RINDEX state of the
    materials and the tables kept by the physics engine."""
    import opengate_core as g4

    pe = simulation_engine.physics_engine
    info = {}
    for mat in g4.G4Material.GetMaterialTable:
        name = str(mat.GetName())
        mpt = mat.GetMaterialPropertiesTable()
        if mpt is None:
            info[name] = None
        else:
            values = mpt.GetPropertyValues("RINDEX")
            info[name] = None if values is None else list(values[1])
    simulation_engine.user_hook_log["rindex"] = info
    simulation_engine.user_hook_log["tables"] = sorted(pe.g4_xray_rindex_tables)


def make_sim(paths, phase_contrast, energy=20 * keV, data_dir=None, image=False):
    sim = gate.Simulation()
    sim.output_dir = paths.output
    sim.number_of_threads = 1
    sim.visu = False
    sim.random_seed = 123456
    sim.world.size = [30 * cm, 30 * cm, 30 * cm]
    sim.world.material = "G4_AIR"
    sim.physics_manager.xray_phase_contrast = phase_contrast
    if data_dir is not None:
        sim.physics_manager.xray_refractive_index_data_dir = data_dir

    for i in range(2 if image else 1):
        box = sim.add_volume("Box", f"box{i}")
        box.size = [2 * cm, 2 * cm, 2 * cm]
        box.translation = [(i - 0.5) * 6 * cm, 0, 0]
        box.material = "G4_WATER"

    if image:
        array = np.zeros((2, 2, 2), dtype=np.float32)
        array[0, 0, 0] = 1
        img = itk.image_from_array(array)
        img.SetSpacing([5.0, 5.0, 5.0])
        fn = paths.output / "test140_labels.mhd"
        itk.imwrite(img, str(fn))
        vol = sim.add_volume("Image", "vox")
        vol.image = fn
        vol.material = "G4_AIR"
        vol.translation = [0, 6 * cm, 0]
        vol.voxel_materials = [[-0.5, 0.5, "G4_WATER"], [0.5, 1.5, "G4_PLEXIGLASS"]]

    source = sim.add_source("GenericSource", "src")
    source.particle = "gamma"
    source.energy.mono = energy
    source.position.type = "point"
    source.position.translation = [0, 0, -10 * cm]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.number_of_primaries = 10
    sim.user_hook_after_init = log_materials
    return sim


def run(sim):
    # Geant4 materials persist within a process: each scenario runs in its own
    sim.run(start_new_process=True)
    return sim.user_hook_log


def water_rindex(log):
    return np.array(log["rindex"]["G4_WATER"])


if __name__ == "__main__":
    paths = utility.get_default_test_paths(__file__, output_folder="test140_sim")
    is_ok = True

    # --- switch off: no material gets a property table
    log = run(make_sim(paths, False))
    ok = log["rindex"]["G4_WATER"] is None and log["rindex"]["G4_AIR"] is None
    ok = ok and log["tables"] == []
    utility.print_test(ok, "switch off: no RINDEX, no tables")
    is_ok &= ok

    # --- switch on
    log = run(make_sim(paths, True))
    ok = log["rindex"]["G4_WATER"] is not None and log["rindex"]["G4_AIR"] is not None
    ok = ok and log["tables"] == ["G4_AIR", "G4_WATER"]
    utility.print_test(ok, f"switch on: tables = {log['tables']}")
    is_ok &= ok
    default_water = water_rindex(log)

    # --- repeated materials (two boxes + voxel materials)
    log = run(make_sim(paths, True, image=True))
    ok = log["tables"] == ["G4_AIR", "G4_PLEXIGLASS", "G4_WATER"]
    ok = ok and log["rindex"]["G4_PLEXIGLASS"] is not None
    utility.print_test(ok, f"repeated materials: tables = {log['tables']}")
    is_ok &= ok

    # --- own data folder: delta of O doubled
    own = paths.output / "own_data"
    if own.exists():
        shutil.rmtree(own)
    from opengate.xray_refractive_index import DEFAULT_REFRACTIVE_INDEX_DATA_DIR

    shutil.copytree(DEFAULT_REFRACTIVE_INDEX_DATA_DIR, own)
    lines = (own / "O.txt").read_text().splitlines()
    out = []
    for line in lines:
        w = line.split()
        try:
            e, d = float(w[0]), float(w[1])
            out.append(" ".join([w[0], repr(2 * d)] + w[2:]))
        except (ValueError, IndexError):
            out.append(line)
    (own / "O.txt").write_text("\n".join(out) + "\n")
    log = run(make_sim(paths, True, data_dir=str(own)))
    own_water = water_rindex(log)
    ok = not np.allclose(own_water, default_water, rtol=0, atol=1e-12)
    ok = ok and np.all(own_water < default_water)
    utility.print_test(
        ok, f"own folder: max |dRINDEX| = {np.max(np.abs(own_water-default_water))}"
    )
    is_ok &= ok

    # --- above 150 keV: one warning
    sim = make_sim(paths, True, energy=200 * keV)
    run(sim)
    warns = [w for w in sim.warnings if "150 keV" in w]
    ok = len(warns) == 1
    utility.print_test(ok, f"200 keV: {len(warns)} warning(s) with '150 keV'")
    is_ok &= ok
    sim = make_sim(paths, True, energy=20 * keV)
    run(sim)
    ok = not [w for w in sim.warnings if "150 keV" in w]
    utility.print_test(ok, "20 keV: no warning")
    is_ok &= ok

    utility.test_ok(is_ok)
