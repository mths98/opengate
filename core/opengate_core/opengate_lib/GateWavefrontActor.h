/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#ifndef GateWavefrontActor_h
#define GateWavefrontActor_h

#include "GateVActor.h"
#include <G4Cache.hh>
#include <pybind11/stl.h>

namespace py = pybind11;

class G4Material;

/*
 * Complex wavefront of x-ray photons on a virtual plane z = plane_z.
 *
 * Each photon accumulates the phase Phi = sum of -2 pi delta L / lambda
 * along its steps, with delta = 1 - RINDEX(E) of the pre-step material and
 * lambda = h c / E. The first crossing of the plane decides the track: if it
 * is in the +z direction, sqrt(E/keV) cos(Phi), sqrt(E/keV) sin(Phi) and
 * E/keV are added to the pixel hit by the photon; if it is in the -z
 * direction (secondaries born after the plane) the photon is only counted
 * in n_backward_crossings and never summed. The later crossings of the
 * track, in any direction, are recrossings.
 *
 * The three grids (ny x nx, row = y) are shared by all threads and the sums
 * are done under a mutex. The per-photon state (phase, already counted) is
 * thread local and reset at the start of each track.
 */
class GateWavefrontActor : public GateVActor {

public:
  // Constructor
  GateWavefrontActor(py::dict &user_info);

  void InitializeUserInfo(py::dict &user_info) override;

  // Allocate the grids (once: they are accumulated over the runs)
  void InitializeCpp() override;

  // Only creates the thread local state
  void BeginOfRunAction(const G4Run *) override;

  void PreUserTrackingAction(const G4Track *) override;

  // Main function called every step in attached volume
  void SteppingAction(G4Step *) override;

  // Counters: number of photons counted, outside the grid, recrossings,
  // backward (-z) crossings of photons not yet counted and sum of the
  // energies of the counted photons (keV)
  py::dict GetCounters() const;

  int GetSizeX() const { return fSizeX; }
  int GetSizeY() const { return fSizeY; }

  // Grids (ny x nx, row = y), exposed without copy to python
  std::vector<double> fRealGrid;
  std::vector<double> fImagGrid;
  std::vector<double> fEnergyGrid;

protected:
  G4double GetRefractiveIndex(const G4Material *material,
                              G4double energy) const;

  G4double fPlaneZ = 0;
  G4double fCenterX = 0;
  G4double fCenterY = 0;
  int fSizeX = 0;
  int fSizeY = 0;
  G4double fSpacingX = 0;
  G4double fSpacingY = 0;
  G4bool fKillAfterPlane = false;
  // a step ending closer than this to the plane crosses it
  G4double fPlaneTolerance = 0;

  long fNumberOfCounted = 0;
  long fNumberOutsideGrid = 0;
  long fNumberOfRecrossings = 0;
  long fNumberOfBackward = 0;
  G4double fSumEnergy = 0; // keV

  struct threadLocalWavefront {
    G4double fPhase = 0;
    G4bool fCounted = false;
  };
  G4Cache<threadLocalWavefront> fThreadLocalData;
};

#endif // GateWavefrontActor_h
