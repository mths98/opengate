#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import re

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
    overrides = xri.read_reference_densities()
    rho_i = [
        overrides.get(s, nist.FindOrBuildMaterial("G4_" + s).GetDensity() / g_cm3)
        for s in symbols
    ]
    delta = xri.compound_delta(
        symbols, fractions, material.GetDensity() / g_cm3, rho_i, grid_keV
    )
    return 1.0 - delta


def test_negative_delta_materials():
    """Materials with elements having delta < 0 near the L/M edges."""
    nist = g4.G4NistManager.Instance()
    grid = xri.default_energy_grid_keV()
    db = MaterialDatabase()
    db.read_from_file(f"{contrib_path[0]}/GateMaterials.db")
    ok = True
    # LYSO is not a NIST material of Geant4: it comes from the GATE database
    for name, finder in [
        ("G4_GADOLINIUM_OXYSULFIDE", nist),
        ("G4_BGO", nist),
        ("LYSO", db),
    ]:
        mat = finder.FindOrBuildMaterial(name)
        try:
            apply_xray_rindex([mat])
        except Exception as e:
            utility.print_test(False, f"{name}: raised {e}")
            ok = False
            continue
        values = mat.GetMaterialPropertiesTable().GetPropertyValues("RINDEX")
        rindex = np.array(values[1])
        diff = np.max(np.abs(rindex - expected_rindex(mat, grid)))
        n_above_one = int(np.sum(rindex > 1.0))
        this_ok = bool(diff <= 1e-12 and np.all(np.isfinite(rindex)))
        utility.print_test(
            this_ok, f"{name}: max |RINDEX - (1-delta)|={diff}, RINDEX>1 at {n_above_one} nodes"
        )
        ok = ok and this_ok
    return ok


def test_foreign_mpt_swapped_after():
    """A foreign MPT attached after our call is detected (identity check)."""
    nist = g4.G4NistManager.Instance()
    mat = nist.FindOrBuildMaterial("G4_Cu")
    apply_xray_rindex([mat])
    foreign = g4.G4MaterialPropertiesTable()
    foreign.AddProperty("RINDEX", [1 * keV, 150 * keV], [1.5, 1.5], False, False)
    mat.SetMaterialPropertiesTable(foreign)
    try:
        apply_xray_rindex([mat])
    except Exception as e:
        ok = "G4_Cu" in str(e)
        utility.print_test(ok, f"foreign MPT swapped after our call -> {e}")
        return ok
    utility.print_test(False, "foreign MPT swapped after our call did not raise")
    return False


def test_implicit_densities():
    """Density implied by delta at 300 keV vs the density used in Eq. 21.

    delta ~ r_e lambda^2 rho N_A Z / (2 pi A), with f1 ~ Z. Fails if the ratio
    is outside [0.5, 2]; lists the elements further than 3 % from 1.
    """
    nist = g4.G4NistManager.Instance()
    overrides = xri.read_reference_densities()
    r_e = 2.8179403262e-13  # cm
    n_avogadro = 6.02214076e23
    hc = 12.398419843  # keV * Angstrom
    lam = hc / 300.0 * 1e-8  # cm
    symbols = sorted(
        f.stem
        for f in pathlib.Path(xri.DEFAULT_REFRACTIVE_INDEX_DATA_DIR).glob("*.txt")
        if re.fullmatch(r"[A-Z][a-z]?", f.stem)
    )
    outside = []
    ratios = {}
    for symbol in symbols:
        pure = nist.FindOrBuildMaterial("G4_" + symbol)
        element = pure.GetElement(0)
        z = element.GetZ()
        a = element.GetA() / (g4_units.g / g4_units.mole)
        energies, delta = xri.read_element_delta(symbol)
        d300 = float(xri.interpolate_delta(energies, delta, 300.0, label=symbol))
        rho_implied = d300 * 2 * np.pi * a / (r_e * lam**2 * n_avogadro * z)
        rho_used = overrides.get(symbol, pure.GetDensity() / g_cm3)
        ratios[symbol] = rho_implied / rho_used
        if abs(ratios[symbol] - 1.0) > 0.03:
            outside.append(symbol)
    print("  element   implied/used - 1   (outside +-3 %)")
    for symbol in outside:
        print(f"  {symbol:7s}   {100 * (ratios[symbol] - 1):+9.1f} %")
    worst = max(ratios, key=lambda k: abs(np.log(ratios[k])))
    ok = all(0.5 <= r <= 2.0 for r in ratios.values())
    utility.print_test(
        ok,
        f"{len(symbols)} elements, implied density within [0.5, 2] of the used "
        f"one (worst: {worst} {ratios[worst]:.3f}); {len(outside)} outside 3 %",
    )
    # Br uses the liquid density through reference_densities.txt
    br_density = overrides.get("Br", 0)
    ok_br = abs(br_density - 3.1028) < 1e-12 and abs(ratios["Br"] - 1) < 0.03
    utility.print_test(ok_br, f"Br: override {br_density}, ratio {ratios['Br']:.3f}")
    return ok and ok_br


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
        ref_db = expected_rindex(gate_water, grid)
        d_db = np.max(np.abs(np.array(values[1]) - ref_db))
        ok_db = rel <= 0.01 and d_db <= 1e-12
        utility.print_test(d_db <= 1e-12, f"GATE Water vs Eq. 21: max dRINDEX={d_db}")
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

    # --- foreign RINDEX (e.g. optical) on a fresh material: fatal
    ok_fatal = False
    fe = nist.FindOrBuildMaterial("G4_Fe")
    foreign = g4.G4MaterialPropertiesTable()
    foreign.AddProperty("RINDEX", [1 * keV, 150 * keV], [1.5, 1.5], False, False)
    fe.SetMaterialPropertiesTable(foreign)
    try:
        apply_xray_rindex([fe])
    except Exception as e:
        ok_fatal = "G4_Fe" in str(e)
        utility.print_test(ok_fatal, f"foreign RINDEX -> {e}")
    if not ok_fatal:
        utility.print_test(False, "foreign RINDEX did not raise")

    # --- re-applying our own RINDEX replaces it (successive simulations)
    ok_again = False
    try:
        grid2 = np.geomspace(1.0, 150.0, 100)
        apply_xray_rindex([water], energy_grid_keV=grid2)
        v2 = water.GetMaterialPropertiesTable().GetPropertyValues("RINDEX")
        ref2 = expected_rindex(water, grid2)
        ok_again = (
            len(v2[0]) == 100
            and np.max(np.abs(np.array(v2[1]) - ref2)) <= 1e-12
            and np.max(np.abs(np.array(v2[0]) - grid2 * keV)) <= 1e-9
        )
    except Exception as e:
        utility.print_test(False, f"re-apply raised: {e}")
    utility.print_test(ok_again, "re-apply on same material: replaced, 100 nodes")

    # --- one entry per material
    al = nist.FindOrBuildMaterial("G4_Al")
    out = apply_xray_rindex([al, nist.FindOrBuildMaterial("G4_Cu")])
    ok_dict = sorted(out.keys()) == ["G4_Al", "G4_Cu"]
    utility.print_test(ok_dict, f"returned keys: {sorted(out.keys())}")

    ok_neg = test_negative_delta_materials()
    ok_swap = test_foreign_mpt_swapped_after()
    ok_dens = test_implicit_densities()

    utility.test_ok(
        is_ok
        and ok_db
        and ok_mpt
        and ok_fatal
        and ok_again
        and ok_dict
        and ok_neg
        and ok_swap
        and ok_dens
    )
