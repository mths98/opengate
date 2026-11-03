#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate_core as g4
import opengate.tests.utility as utility
from opengate.utility import g4_units

if __name__ == "__main__":
    utility.print_test(True, "Read back vector properties of a material table")
    keV = g4_units.keV

    mpt = g4.G4MaterialPropertiesTable()
    energies = [1 * keV, 2 * keV]
    values = [0.9, 0.8]
    mpt.AddProperty("RINDEX", energies, values, False, False)

    result = mpt.GetPropertyValues("RINDEX")
    is_ok = result is not None
    if is_ok:
        read_energies, read_values = result
        is_ok = list(read_energies) == energies and list(read_values) == values
    utility.print_test(is_ok, f"RINDEX read back: {result}")

    # a known key that was never set
    missing = mpt.GetPropertyValues("ABSLENGTH")
    ok_missing = missing is None
    utility.print_test(ok_missing, f"ABSLENGTH not set -> {missing}")

    # a key unknown to Geant4: also None, no exception
    unknown = mpt.GetPropertyValues("NOT_A_GEANT4_KEY")
    ok_unknown = unknown is None
    utility.print_test(ok_unknown, f"unknown key -> {unknown}")

    utility.test_ok(is_ok and ok_missing and ok_unknown)
