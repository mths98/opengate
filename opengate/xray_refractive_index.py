"""Reading and interpolation of the X-ray refractive index decrement (delta).

The refractive index of a material is n = 1 - delta + i*beta. The tables in
``opengate/data/complex_refractive_index/`` give, for each element, the energy
(keV), delta and beta. Only delta is read here.
"""

import pathlib

import numpy as np

import opengate_core as g4

from .exception import fatal, warning
from .utility import g4_units

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


def compound_delta(
    symbols,
    mass_fractions,
    density_g_cm3,
    element_densities_g_cm3,
    energy_grid_keV,
    data_dir=None,
):
    """Delta of a compound on an energy grid, by the mass-fraction mixing rule.

    delta(E) = rho * sum_i w_i * delta_i(E) / rho_i   (Eq. 21)

    where w_i is the mass fraction and rho_i the density of the pure element i,
    and rho the density of the compound (g/cm3). The mass fractions are not
    renormalized. The element tables are interpolated log-log on the grid.
    """
    n = len(symbols)
    if len(mass_fractions) != n or len(element_densities_g_cm3) != n:
        fatal(
            "X-ray refractive index: symbols, mass fractions and element "
            f"densities must have the same length (got {n}, "
            f"{len(mass_fractions)}, {len(element_densities_g_cm3)})."
        )
    if n == 0:
        fatal("X-ray refractive index: a compound needs at least one element.")
    if not density_g_cm3 > 0:
        fatal(
            "X-ray refractive index: the compound density must be positive "
            f"(got {density_g_cm3})."
        )
    grid = np.asarray(energy_grid_keV, dtype=float)
    result = np.zeros_like(grid)
    for symbol, w, rho_i in zip(symbols, mass_fractions, element_densities_g_cm3):
        if not rho_i > 0:
            fatal(
                f"X-ray refractive index: the density of element '{symbol}' "
                f"must be positive (got {rho_i})."
            )
        energies, delta = read_element_delta(symbol, data_dir)
        result += w * interpolate_delta(energies, delta, grid, label=symbol) / rho_i
    return density_g_cm3 * result


def apply_xray_rindex(g4_materials, data_dir=None, energy_grid_keV=None):
    """Store RINDEX = 1 - delta in the property table of each G4Material.

    delta follows the mass-fraction mixing rule (Eq. 21), computed from the
    elements of the material (G4Element symbol, not name) and the density of
    each pure element (the NIST material G4_<symbol>).

    If a material has no property table, one is created. If it has one without
    RINDEX, RINDEX is added to it. If it already has RINDEX, this is fatal.
    Returns a dict {material name: property table}; the caller must keep it so
    that the tables stay alive.
    """
    grid = default_energy_grid_keV() if energy_grid_keV is None else energy_grid_keV
    grid = np.asarray(grid, dtype=float)
    keV = g4_units.keV
    g_cm3 = g4_units.g / g4_units.cm3
    nist = g4.G4NistManager.Instance()
    tables = {}
    for mat in g4_materials:
        name = str(mat.GetName())
        mpt = mat.GetMaterialPropertiesTable()
        if mpt is None:
            mpt = g4.G4MaterialPropertiesTable()
        elif mpt.GetPropertyValues("RINDEX") is not None:
            fatal(
                f"X-ray refractive index: material '{name}' already has a "
                "RINDEX property in its material properties table."
            )
        n = mat.GetNumberOfElements()
        symbols = [str(mat.GetElement(i).GetSymbol()) for i in range(n)]
        fractions = [mat.GetElementFraction(i) for i in range(n)]
        rho_i = []
        for symbol in symbols:
            pure = nist.FindOrBuildMaterial("G4_" + symbol)
            if pure is None:
                fatal(
                    f"X-ray refractive index: cannot find the NIST material "
                    f"'G4_{symbol}' to get the density of element '{symbol}' "
                    f"(material '{name}')."
                )
            rho_i.append(pure.GetDensity() / g_cm3)
        delta = compound_delta(
            symbols, fractions, mat.GetDensity() / g_cm3, rho_i, grid, data_dir
        )
        mpt.AddProperty("RINDEX", list(grid * keV), list(1.0 - delta), False, False)
        mat.SetMaterialPropertiesTable(mpt)
        tables[name] = mpt
    return tables
