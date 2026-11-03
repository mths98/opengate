"""Reading and interpolation of the X-ray refractive index decrement (delta).

The refractive index of a material is n = 1 - delta + i*beta. The tables in
``opengate/data/complex_refractive_index/`` give, for each element, the energy
(keV), delta and beta. Only delta is read here.
"""

import pathlib

import numpy as np

from .exception import fatal, warning

DEFAULT_REFRACTIVE_INDEX_DATA_DIR = (
    pathlib.Path(__file__).resolve().parent / "data" / "complex_refractive_index"
)

# cache of the tables already read: (symbol, resolved folder) -> (E, delta)
_element_delta_cache = {}

# (label, side) pairs for which an extrapolation warning was already issued
_extrapolation_warned = set()


def _reset_extrapolation_warnings():
    """Forget the extrapolation warnings already issued (used by the tests)."""
    _extrapolation_warned.clear()


def default_energy_grid_keV():
    """Default energy grid (keV) on which delta is tabulated for a material."""
    return np.geomspace(1.0, 150.0, 500)


def read_element_delta(symbol, data_dir=None):
    """Read the table of an element and return (energies in keV, delta).

    The file ``<symbol>.txt`` is searched in data_dir, or in the folder shipped
    with the package if data_dir is None. It is a fatal error if it is missing.
    """
    folder = (
        DEFAULT_REFRACTIVE_INDEX_DATA_DIR
        if data_dir is None
        else pathlib.Path(data_dir)
    )
    folder = folder.expanduser().resolve()
    key = (symbol, str(folder))
    if key in _element_delta_cache:
        return _element_delta_cache[key]

    path = folder / f"{symbol}.txt"
    if not path.is_file():
        fatal(
            f"X-ray refractive index: no data file for element '{symbol}'. "
            f"Expected '{path}' in the folder '{folder}'."
        )
    table = np.loadtxt(path, ndmin=2)
    if table.shape[1] < 2:
        fatal(f"X-ray refractive index: file '{path}' must have at least 2 columns.")
    energies = np.ascontiguousarray(table[:, 0], dtype=float)
    delta = np.ascontiguousarray(table[:, 1], dtype=float)
    energies.setflags(write=False)
    delta.setflags(write=False)
    _element_delta_cache[key] = (energies, delta)
    return energies, delta


def _warn_extrapolation(label, side, e_requested, energies, edge_value):
    key = (label, side)
    if key in _extrapolation_warned:
        return
    _extrapolation_warned.add(key)
    warning(
        f"X-ray refractive index: energy {e_requested:.3f} keV is {side} the "
        f"tabulated range [{energies[0]:.3f}, {energies[-1]:.3f}] keV for "
        f"'{label}'; using the edge value delta={edge_value:.6e}"
    )


def interpolate_delta(energies_keV, delta, e_keV, label=None):
    """Log-log interpolation of delta at e_keV, clamped outside the table.

    A warning is issued once per (label, side) when a requested energy lies
    outside the tabulated range.
    """
    energies = np.asarray(energies_keV, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if np.any(delta <= 0):
        fatal(
            "X-ray refractive index: delta must be strictly positive for the "
            f"log-log interpolation (table '{label}')."
        )
    e = np.asarray(e_keV, dtype=float)
    if label is None:
        label = "table"
    e_min, e_max = float(np.min(e)), float(np.max(e))
    if e_min < energies[0]:
        _warn_extrapolation(label, "below", e_min, energies, delta[0])
    if e_max > energies[-1]:
        _warn_extrapolation(label, "above", e_max, energies, delta[-1])

    e_clamped = np.clip(e, energies[0], energies[-1])
    result = np.exp(np.interp(np.log(e_clamped), np.log(energies), np.log(delta)))
    # exact edge values (avoid round-off of exp(log(x)))
    result = np.where(e <= energies[0], delta[0], result)
    result = np.where(e >= energies[-1], delta[-1], result)
    return result
