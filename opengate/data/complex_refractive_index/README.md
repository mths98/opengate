# Complex refractive index tables

Tables of the complex refractive index n = 1 - delta + i*beta used for X-ray
phase contrast (propagation-based imaging).

## Format

Plain text, three columns separated by spaces or tabs (scientific notation
with `e`/`E` is accepted):

    E [keV]    delta    beta

## Origin

The tables come from the PCI-GAMOS data set (Sanchez-Lara et al., Physica
Medica 142 (2026) 105716). The delta values of the elements (`H.txt`,
`C.txt`, ...) were computed at the NIST density of the pure element.

## Which files are used

Only the element files (named after the element symbol) are used in the
calculation: the delta of a material is obtained from the elemental tables.

The compound files (`Water`, `H2O`, `Water_new`, `PMMA`, `Breast_*`,
`Muscle`, `Glandular`, `Cancerous_mass`, `Skin`, `Blood`, `Ligament`,
`Calcification`, `Compression_paddle`, `AAPM_PMMA`, `SiO3`, `aSe`, `AIR`,
`G4_AIR`, `Pure_Phase`, `Pure_Absorption`) are **not** used in the
calculation. They are kept only as a reference for validation.

## Divergences from the PCI-GAMOS code

- **D17 - negative delta near absorption edges.** The tables of As, Dy, Er,
  Eu, Ga, Gd, Ge, Hf, Ho, Lu, Pm, Sm, Tb, Tm, Yb and Zn have delta < 0 between
  1.0 and 1.7 keV (anomalous dispersion close to the L/M edges). This is real
  physics (RINDEX > 1), so it is accepted. The interpolation is log-log in the
  segments where both nodes have delta > 0 and linear in delta against log(E)
  in the other segments. A delta <= 0 in the result of Eq. 21 is not an error.
- **D18 - reference densities.** Eq. 21 divides by the density of the pure
  element at which its table was computed. This is the density of the Geant4
  material `G4_<symbol>`, except for Br: `Br.txt` was computed for liquid
  bromine (3.1028 g/cm3) while `G4_Br` is a gas (0.00707 g/cm3), which made
  delta about 440 times too large. `reference_densities.txt` overrides the
  density for the elements it lists (only Br). The file is searched in the
  data folder in use; without it there is no override. A consistency test
  estimates the density implied by each table at 300 keV
  (delta = r_e lambda^2 rho N_A Z / (2 pi A), f1 ~ Z). The elements that
  differ more than 3 % from the density used are At (-25 %), Fr (+88 %),
  Ra (+10 %), Rn (+8 %) and Yb (+4 %). These are probably due to the f1 ~ Z
  approximation at high Z or to uncertain NIST densities of rare elements,
  and are kept as a known uncertainty.
