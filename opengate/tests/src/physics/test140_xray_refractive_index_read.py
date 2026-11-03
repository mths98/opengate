#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from opengate.logger import logger
import opengate.tests.utility as utility
from opengate.xray_refractive_index import (
    DEFAULT_REFRACTIVE_INDEX_DATA_DIR,
    read_element_delta,
    interpolate_delta,
    default_energy_grid_keV,
    _reset_extrapolation_warnings,
)


class WarningCapture:
    """Collect the messages sent to the GATE logger at WARNING level."""

    def __enter__(self):
        self.messages = []
        self.handler_id = logger.add(
            lambda m: self.messages.append(m.record["message"]), level="WARNING"
        )
        return self

    def __exit__(self, *args):
        logger.remove(self.handler_id)


def test_hydrogen():
    E, d = read_element_delta("H")
    ok = len(E) == 91 and len(d) == 91
    ok = ok and np.isclose(E[0], 1.067676, rtol=1e-6)
    ok = ok and np.isclose(E[-1], 432.9451, rtol=1e-6)
    v = interpolate_delta(E, d, 20.11215)
    ok = ok and np.isclose(v, 8.535362818253613827e-11, rtol=1e-6)
    utility.print_test(ok, f"H.txt: {len(E)} rows, E=[{E[0]}, {E[-1]}], delta(20.11215)={v}")
    return ok


def test_tab_format():
    E, d = read_element_delta("PMMA")
    ok = len(E) == 500 and np.isclose(E[0], 5)
    E2, d2 = read_element_delta("G4_AIR")
    ok = ok and len(d2) > 0 and np.all(d2 == 0)
    utility.print_test(ok, f"tab/E-notation: PMMA {len(E)} rows, G4_AIR all zero")
    return ok


def test_missing_element():
    try:
        read_element_delta("Xx")
        ok = False
        msg = ""
    except Exception as e:
        msg = str(e)
        ok = "Xx" in msg and str(DEFAULT_REFRACTIVE_INDEX_DATA_DIR) in msg
    utility.print_test(ok, f"missing element raises with symbol and folder: {msg[:120]}")
    return ok


def test_nodes_and_clamp():
    E, d = read_element_delta("O")
    ok = np.allclose(interpolate_delta(E, d, E), d, rtol=1e-12, atol=0)
    with WarningCapture():
        lo = interpolate_delta(E, d, 0.5)
        hi = interpolate_delta(E, d, 1000.0)
    ok = ok and np.isclose(lo, d[0], rtol=1e-12) and np.isclose(hi, d[-1], rtol=1e-12)
    utility.print_test(ok, "interpolation at nodes and clamp outside the range")
    return ok


def test_extrapolation_warning():
    _reset_extrapolation_warnings()
    E, d = read_element_delta("H")
    with WarningCapture() as w:
        interpolate_delta(E, d, 0.5, label="H")
        n1 = len(w.messages)
        m1 = w.messages[0] if w.messages else ""
        interpolate_delta(E, d, 1000.0, label="H")
        n2 = len(w.messages)
        m2 = w.messages[-1] if n2 > n1 else ""
        interpolate_delta(E, d, 0.4, label="H")
        n3 = len(w.messages)
        interpolate_delta(E, d, 20.0, label="H")
        n4 = len(w.messages)
    ok = n1 == 1 and "'H'" in m1 and "below" in m1 and "1.068" in m1 and "432.945" in m1
    ok = ok and n2 == 2 and "above" in m2
    ok = ok and n3 == 2 and n4 == 2
    utility.print_test(ok, f"warning once per (label, side): counts {n1},{n2},{n3},{n4}; '{m1}'")
    return ok


def test_loglog():
    E = np.array([1.0, 10.0, 100.0])
    d = 1e-6 * E**-2
    x = 31.6227766
    v = interpolate_delta(E, d, x)
    ok = np.isclose(v, 1e-6 * x**-2, rtol=1e-10)
    utility.print_test(ok, f"log-log interpolation: {v} vs {1e-6 * x**-2}")
    return ok


def test_energy_grid():
    g = default_energy_grid_keV()
    r = g[1:] / g[:-1]
    ok = len(g) == 500 and np.isclose(g[0], 1.0) and np.isclose(g[-1], 150.0)
    ok = ok and np.allclose(r, r[0], rtol=1e-9)
    utility.print_test(ok, "default energy grid: 500 points, 1-150 keV, constant ratio")
    return ok


def test_custom_dir(paths):
    custom = paths.output / "test140_custom_dir"
    custom.mkdir(parents=True, exist_ok=True)
    (custom / "H.txt").write_text("1 1e-6 0\n100 1e-10 0\n")
    E, d = read_element_delta("H", custom)
    ok = np.allclose(E, [1, 100]) and np.allclose(d, [1e-6, 1e-10], rtol=1e-12)
    E2, d2 = read_element_delta("H")
    ok = ok and len(E2) == 91
    E3, d3 = read_element_delta("H", str(custom))
    ok = ok and len(E3) == 2
    utility.print_test(ok, "custom data dir used, cache does not mix folders")
    return ok


if __name__ == "__main__":
    paths = utility.get_default_test_paths(__file__, output_folder="test140")
    is_ok = True
    is_ok = test_hydrogen() and is_ok
    is_ok = test_tab_format() and is_ok
    is_ok = test_missing_element() and is_ok
    is_ok = test_nodes_and_clamp() and is_ok
    is_ok = test_extrapolation_warning() and is_ok
    is_ok = test_loglog() and is_ok
    is_ok = test_energy_grid() and is_ok
    is_ok = test_custom_dir(paths) and is_ok
    utility.test_ok(is_ok)
