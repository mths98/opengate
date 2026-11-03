/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include "GateXRayRefractionPhysics.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_GateXRayRefractionPhysics(py::module &m) {
  // G4VModularPhysicsList owns the constructors registered with
  // RegisterPhysics and deletes them with the physics list: Python must not
  // delete it as well.
  py::class_<GateXRayRefractionPhysics, G4VPhysicsConstructor,
             std::unique_ptr<GateXRayRefractionPhysics, py::nodelete>>(
      m, "GateXRayRefractionPhysics")
      .def(py::init<G4double>(),
           // The constructor only stores the roughness; the process is
           // created later, in C++, by ConstructProcess.
           py::call_guard<py::gil_scoped_release>(),
           py::arg("roughness_sigma") = 0.0);
}
