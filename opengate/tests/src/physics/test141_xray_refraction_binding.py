#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate_core as g4
import opengate.tests.utility as utility

if __name__ == "__main__":
    utility.print_test(True, "Build the x-ray refraction physics constructor")

    is_ok = True
    for sigma in (0.0, 0.05):
        try:
            physics = g4.GateXRayRefractionPhysics(sigma)
            ok = isinstance(physics, g4.G4VPhysicsConstructor)
            msg = f"sigma = {sigma} rad -> {type(physics).__name__}"
        except Exception as e:
            ok = False
            msg = f"sigma = {sigma} rad -> {type(e).__name__}: {e}"
        utility.print_test(ok, msg)
        is_ok = is_ok and ok

    utility.test_ok(is_ok)
