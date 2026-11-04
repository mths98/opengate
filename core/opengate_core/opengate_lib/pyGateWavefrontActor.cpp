/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "GateWavefrontActor.h"

namespace {
// numpy view (ny, nx) on a grid of the actor, without copy. The base of the
// array is the python actor object: the actor (and thus the grid memory) is
// kept alive as long as the array exists.
py::array_t<double> GridView(py::object self, std::vector<double> &grid,
                             const GateWavefrontActor &actor) {
  const auto nx = static_cast<py::ssize_t>(actor.GetSizeX());
  const auto ny = static_cast<py::ssize_t>(actor.GetSizeY());
  const auto itemsize = static_cast<py::ssize_t>(sizeof(double));
  return py::array_t<double>({ny, nx}, {nx * itemsize, itemsize}, grid.data(),
                             self);
}
} // namespace

void init_GateWavefrontActor(py::module &m) {
  py::class_<GateWavefrontActor,
             std::unique_ptr<GateWavefrontActor, py::nodelete>, GateVActor>(
      m, "GateWavefrontActor")
      .def(py::init<py::dict &>())
      .def("GetRealGrid",
           [](py::object self) {
             auto &a = self.cast<GateWavefrontActor &>();
             return GridView(self, a.fRealGrid, a);
           })
      .def("GetImagGrid",
           [](py::object self) {
             auto &a = self.cast<GateWavefrontActor &>();
             return GridView(self, a.fImagGrid, a);
           })
      .def("GetEnergyGrid",
           [](py::object self) {
             auto &a = self.cast<GateWavefrontActor &>();
             return GridView(self, a.fEnergyGrid, a);
           })
      .def("GetCounters", &GateWavefrontActor::GetCounters);
}
