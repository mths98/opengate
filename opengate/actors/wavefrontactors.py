import json
from pathlib import Path

import itk

import opengate_core as g4

from ..base import process_cls
from ..exception import fatal
from ..utility import g4_units
from .actoroutput import ActorOutputBase
from .base import ActorBase


class ActorOutputWavefrontActor(ActorOutputBase):
    """Output of the WavefrontActor: the real, imaginary and energy grids,
    the counters, and the names of the files written on disk."""

    # hints for IDE
    output_filename: str
    write_to_disk: bool

    user_info_defaults = {
        "output_filename": (
            "wavefront.mhd",
            {
                "doc": "Base name of the output files: <base>_real.mhd, "
                "<base>_imag.mhd, <base>_energy.mhd and <base>.json. "
                "Relative to the simulation output directory.",
            },
        ),
        "write_to_disk": (
            True,
            {"doc": "Should the grids and the JSON file be written on disk?"},
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real = None
        self.imag = None
        self.energy = None
        self.counters = {}

    def get_output_filename(self, **kwargs):
        return self.output_filename

    def set_output_filename(self, value, **kwargs):
        self.output_filename = value

    def get_write_to_disk(self, **kwargs):
        return self.write_to_disk

    def set_write_to_disk(self, value, **kwargs):
        self.write_to_disk = value

    def get_data(self, which="merged", **kwargs):
        return {
            "real": self.real,
            "imag": self.imag,
            "energy": self.energy,
            "counters": self.counters,
        }


class WavefrontActor(ActorBase, g4.GateWavefrontActor):
    """Complex wavefront of x-ray photons on a virtual plane z = plane_z.

    Each photon accumulates the phase Phi = sum of -2 pi delta L / lambda
    along its steps (delta = 1 - RINDEX(E) of the material, lambda = h c / E).
    At its first crossing of the plane, the pixel hit by the photon receives
    sqrt(E/keV) cos(Phi), sqrt(E/keV) sin(Phi) and E/keV. Only gammas are
    considered. The grids are accumulated over the runs.

    Notes:

    - the phase is only accumulated in the volumes where the actor is
      registered: attach it to the world (the default), or to a mother volume
      containing the whole path of the photons;
    - the phase is not accumulated after the first crossing: the geometry
      after the plane should only contain vacuum or air (not enforced);
    - a photon created on the plane (within the geometrical tolerance) is
      not counted;
    - kill_after_plane also kills the photons crossing outside the grid;
    - in the JSON file, effective_energy_keV is null when no photon is
      counted (N = 0).

    Requires physics_manager.xray_phase_contrast = True, which fills the
    RINDEX of the materials.

    After the simulation, actor.real, actor.imag and actor.energy are numpy
    arrays of shape (ny, nx) (row = y) and actor.counters is a dict.
    """

    # hints for IDE
    plane_z: float
    plane_center: list
    size: list
    spacing: list
    kill_after_plane: bool

    user_info_defaults = {
        "plane_z": (
            None,
            {"doc": "z of the virtual plane, in the world frame (required)."},
        ),
        "plane_center": (
            [0, 0],
            {"doc": "[x, y] of the centre of the grid, in the world frame."},
        ),
        "size": (
            [1000, 1000],
            {"doc": "[nx, ny] number of pixels of the grid."},
        ),
        "spacing": (
            [1 * g4_units.um, 1 * g4_units.um],
            {"doc": "[dx, dy] size of a pixel."},
        ),
        "kill_after_plane": (
            False,
            {"doc": "Kill the photons after their first crossing of the plane."},
        ),
    }

    user_output_config = {
        "wavefront": {
            "actor_output_class": ActorOutputWavefrontActor,
        },
    }

    def __init__(self, *args, **kwargs):
        ActorBase.__init__(self, *args, **kwargs)
        self.__initcpp__()

    def __initcpp__(self):
        g4.GateWavefrontActor.__init__(self, self.user_info)
        self.AddActions(
            {
                "BeginOfRunAction",
                "PreUserTrackingAction",
                "SteppingAction",
                "EndSimulationAction",
            }
        )

    def initialize(self):
        if not self.simulation.physics_manager.xray_phase_contrast:
            fatal(
                f"The WavefrontActor '{self.name}' needs the refractive index "
                "of the materials: set "
                "sim.physics_manager.xray_phase_contrast = True."
            )
        if self.plane_z is None:
            fatal(f"The WavefrontActor '{self.name}' needs plane_z.")
        if len(self.size) != 2 or min(self.size) < 1:
            fatal(
                f"The WavefrontActor '{self.name}': size must be [nx, ny] "
                f"with nx, ny >= 1, got {self.size}."
            )
        if len(self.spacing) != 2 or min(self.spacing) <= 0:
            fatal(
                f"The WavefrontActor '{self.name}': spacing must be [dx, dy] "
                f"with dx, dy > 0, got {self.spacing}."
            )
        if len(self.plane_center) != 2:
            fatal(
                f"The WavefrontActor '{self.name}': plane_center must be "
                f"[x, y], got {self.plane_center}."
            )
        ActorBase.initialize(self)
        self.InitializeUserInfo(self.user_info)
        self.InitializeCpp()

    @property
    def real(self):
        return self.user_output.wavefront.real

    @property
    def imag(self):
        return self.user_output.wavefront.imag

    @property
    def energy(self):
        return self.user_output.wavefront.energy

    @property
    def counters(self):
        return self.user_output.wavefront.counters

    def EndSimulationAction(self):
        # views on the C++ grids, without copy
        output = self.user_output.wavefront
        output.real = self.GetRealGrid()
        output.imag = self.GetImagGrid()
        output.energy = self.GetEnergyGrid()
        output.counters = dict(self.GetCounters())
        if output.write_to_disk:
            self.write_output_files()

    def write_output_files(self):
        """<base>_real.mhd, <base>_imag.mhd, <base>_energy.mhd (2D, double)
        and <base>.json. The ITK origin is the centre of the first pixel."""
        mm = g4_units.mm
        output = self.user_output.wavefront
        path = Path(output.get_output_path())
        base = path.parent / path.stem
        base.parent.mkdir(parents=True, exist_ok=True)
        dx, dy = (float(s) for s in self.spacing)
        nx, ny = (int(s) for s in self.size)
        x0 = self.plane_center[0] - nx * dx / 2
        y0 = self.plane_center[1] - ny * dy / 2
        for suffix, grid in (
            ("real", output.real),
            ("imag", output.imag),
            ("energy", output.energy),
        ):
            img = itk.image_view_from_array(grid)
            img.SetSpacing([dx, dy])
            img.SetOrigin([x0 + dx / 2, y0 + dy / 2])
            itk.imwrite(img, f"{base}_{suffix}.mhd")
        c = output.counters
        n = c["n_photons_counted"]
        meta = {
            "n_photons_counted": n,
            "n_outside_grid": c["n_outside_grid"],
            "n_recrossings": c["n_recrossings"],
            "effective_energy_keV": c["sum_energy_keV"] / n if n > 0 else None,
            "plane_z_mm": self.plane_z / mm,
            "plane_center_mm": [self.plane_center[0] / mm, self.plane_center[1] / mm],
            "grid_corner_mm": [x0 / mm, y0 / mm],
            "mhd_offset_convention": "center of first pixel",
            "size": [nx, ny],
            "spacing_mm": [dx / mm, dy / mm],
            "kill_after_plane": self.kill_after_plane,
        }
        with open(f"{base}.json", "w") as f:
            json.dump(meta, f, indent=2)


process_cls(ActorOutputWavefrontActor)
process_cls(WavefrontActor)
