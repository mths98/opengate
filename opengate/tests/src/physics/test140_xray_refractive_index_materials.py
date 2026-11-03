#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import opengate_core as g4
import opengate.tests.utility as utility
from opengate.utility import g4_units
from opengate.geometry.materials import MaterialDatabase
from opengate.contrib import __path__ as contrib_path
from opengate import xray_refractive_index as xri
from opengate.xray_refractive_index import apply_xray_rindex

keV = g4_units.keV
g_cm3 = g4_units.g / g4_units.cm3


def expected_rindex(material, grid_keV):
    """1 - delta computed independently from the material itself."""
    nist = g4.G4NistManager.Instance()
    n = material.GetNumberOfElements()
    symbols = [str(material.GetElement(i).GetSymbol()) for i in range(n)]
    fractions = [material.GetElementFraction(i) for i in range(n)]
    rho_i = [nist.FindOrBuildMaterial("G4_" + s).GetDensity() / g_cm3 for s in symbols]
    delta = xri.compound_delta(
        symbols, fractions, material.GetDensity() / g_cm3, rho_i, grid_keV
    )
    return 1.0 - delta


if __name__ == "__main__":
    utility.print_test(True, "Write RINDEX in the material property tables")
    nist = g4.G4NistManager.Instance()
    grid = xri.default_energy_grid_keV()

    # --- G4_WATER from NIST
    water = nist.FindOrBuildMaterial("G4_WATER")
    result = apply_xray_rindex([water])
    values = water.GetMaterialPropertiesTable().GetPropertyValues("RINDEX")
    is_ok = values is not None and "G4_WATER" in result and len(result) == 1
    if is_ok:
        energies, rindex = np.array(values[0]), np.array(values[1])
        ref = expected_rindex(water, grid)
        d_e = np.max(np.abs(energies - grid * keV))
        d_v = np.max(np.abs(rindex - ref))
        is_ok = len(energies) == 500 and d_e <= 1e-9 and d_v <= 1e-12
        utility.print_test(is_ok, f"G4_WATER: max dE={d_e}, max dRINDEX={d_v}")
    else:
        utility.print_test(False, "G4_WATER: no RINDEX")

    # --- Water from the GATE database (elements named Hydrogen/Oxygen)
    db = MaterialDatabase()
    db.read_from_file(f"{contrib_path[0]}/GateMaterials.db")
    gate_water = db.FindOrBuildMaterial("Water")
    symbols = [str(gate_water.GetElement(i).GetSymbol()) for i in range(2)]
    utility.print_test(True, f"GATE db Water element symbols: {symbols}")
    result_db = apply_xray_rindex([gate_water])
    values = gate_water.GetMaterialPropertiesTable().GetPropertyValues("RINDEX")
    ok_db = values is not None
    if ok_db:
        e_keV = np.array(values[0]) / keV
        n20 = np.interp(np.log(20.0), np.log(e_keV), np.array(values[1]))
        delta20 = 1.0 - n20
        e_w, d_w = xri.read_element_delta("Water")
        d_w20 = float(xri.interpolate_delta(e_w, d_w, np.array([20.0]))[0])
        rel = abs(delta20 - d_w20) / d_w20
        ok_db = rel <= 0.01
        utility.print_test(ok_db, f"GATE Water delta(20 keV) vs Water.txt: {rel}")
    utility.print_test(ok_db, "GATE db Water has RINDEX")

    # --- existing MPT without RINDEX: ABSLENGTH is kept
    pmma = nist.FindOrBuildMaterial("G4_PLEXIGLASS")
    mpt = g4.G4MaterialPropertiesTable()
    mpt.AddProperty("ABSLENGTH", [1 * keV, 150 * keV], [10.0, 20.0], False, False)
    pmma.SetMaterialPropertiesTable(mpt)
    apply_xray_rindex([pmma])
    mpt_after = pmma.GetMaterialPropertiesTable()
    abs_after = mpt_after.GetPropertyValues("ABSLENGTH")
    ok_mpt = (
        mpt_after.GetPropertyValues("RINDEX") is not None
        and abs_after is not None
        and list(abs_after[1]) == [10.0, 20.0]
    )
    utility.print_test(ok_mpt, "existing MPT: RINDEX added, ABSLENGTH kept")

    # --- existing RINDEX: fatal error citing the material name
    ok_fatal = False
    try:
        apply_xray_rindex([water])
    except Exception as e:
        ok_fatal = "G4_WATER" in str(e)
        utility.print_test(ok_fatal, f"existing RINDEX -> {e}")
    if not ok_fatal:
        utility.print_test(False, "existing RINDEX did not raise")

    # --- one entry per material
    al = nist.FindOrBuildMaterial("G4_Al")
    out = apply_xray_rindex([al, nist.FindOrBuildMaterial("G4_Cu")])
    ok_dict = sorted(out.keys()) == ["G4_Al", "G4_Cu"]
    utility.print_test(ok_dict, f"returned keys: {sorted(out.keys())}")

    utility.test_ok(is_ok and ok_db and ok_mpt and ok_fatal and ok_dict)
