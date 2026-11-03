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
