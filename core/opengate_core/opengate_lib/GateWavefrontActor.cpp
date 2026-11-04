/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include "GateWavefrontActor.h"
#include "GateHelpersDict.h"

#include <G4Gamma.hh>
#include <G4GeometryTolerance.hh>
#include <G4Material.hh>
#include <G4MaterialPropertiesIndex.hh>
#include <G4MaterialPropertiesTable.hh>
#include <G4PhysicalConstants.hh>
#include <G4Step.hh>
#include <G4SystemOfUnits.hh>
#include <G4Track.hh>

#include <algorithm>
#include <cmath>

// Mutex used by the threads to add to the grids and counters (shared by all
// the instances of the actor, as in GateFluenceActor)
static G4Mutex WavefrontGridMutex = G4MUTEX_INITIALIZER;

GateWavefrontActor::GateWavefrontActor(py::dict &user_info)
    : GateVActor(user_info, true) {}

void GateWavefrontActor::InitializeUserInfo(py::dict &user_info) {
  // IMPORTANT: call the base class method
  GateVActor::InitializeUserInfo(user_info);
  fPlaneZ = DictGetDouble(user_info, "plane_z");
  const auto center = DictGetVecDouble(user_info, "plane_center");
  const auto size = DictGetVecInt(user_info, "size");
  const auto spacing = DictGetVecDouble(user_info, "spacing");
  fCenterX = center[0];
  fCenterY = center[1];
  fSizeX = size[0];
  fSizeY = size[1];
  fSpacingX = spacing[0];
  fSpacingY = spacing[1];
  fKillAfterPlane = DictGetBool(user_info, "kill_after_plane");
  fPlaneTolerance = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
}

void GateWavefrontActor::InitializeCpp() {
  GateVActor::InitializeCpp();
  // The grids are allocated once and never reallocated: the python views
  // point to their memory, and the sums accumulate over the runs
  const auto n = static_cast<std::size_t>(fSizeX) * fSizeY;
  fRealGrid.assign(n, 0.0);
  fImagGrid.assign(n, 0.0);
  fEnergyGrid.assign(n, 0.0);
  fNumberOfCounted = 0;
  fNumberOutsideGrid = 0;
  fNumberOfRecrossings = 0;
  fSumEnergy = 0;
}

void GateWavefrontActor::BeginOfRunAction(const G4Run *) {
  // creates the thread local state of this thread
  auto &data = fThreadLocalData.Get();
  data.fPhase = 0;
  data.fCounted = false;
}

void GateWavefrontActor::PreUserTrackingAction(const G4Track *) {
  auto &data = fThreadLocalData.Get();
  data.fPhase = 0;
  data.fCounted = false;
}

G4double GateWavefrontActor::GetRefractiveIndex(const G4Material *material,
                                                G4double energy) const {
  G4MaterialPropertyVector *rindex = nullptr;
  auto *mpt = material->GetMaterialPropertiesTable();
  if (mpt != nullptr) {
    rindex = mpt->GetProperty(kRINDEX);
  }
  if (rindex == nullptr) {
    G4ExceptionDescription ed;
    ed << "The material '" << material->GetName()
       << "' has no RINDEX property. The wavefront actor '" << fActorName
       << "' needs the refractive index of every material crossed by the "
          "photons: it should have been filled by the xray_phase_contrast "
          "option of the physics manager.";
    G4Exception("GateWavefrontActor", "WavefrontActor01", FatalException, ed);
    return 1.0;
  }
  return rindex->Value(energy);
}

void GateWavefrontActor::SteppingAction(G4Step *step) {
  auto *track = step->GetTrack();
  if (track->GetDefinition() != G4Gamma::Definition()) {
    return;
  }
  auto &data = fThreadLocalData.Get();
  const auto *pre = step->GetPreStepPoint();
  const auto *post = step->GetPostStepPoint();

  // 1. does the step cross the plane? A step starting on the plane does not
  // (the previous step, ending on it, did)
  const G4ThreeVector &p0 = pre->GetPosition();
  const G4ThreeVector &p1 = post->GetPosition();
  const G4double d0 = p0.z() - fPlaneZ;
  const G4double d1 = p1.z() - fPlaneZ;
  const G4bool crosses = std::abs(d1) < fPlaneTolerance ||
                         (d0 * d1 < 0 && std::abs(d0) >= fPlaneTolerance);

  // 2. photon already counted: the phase is no longer needed, only the
  // recrossings are recorded (diagnostic)
  if (data.fCounted) {
    if (crosses) {
      G4AutoLock mutex(&WavefrontGridMutex);
      fNumberOfRecrossings++;
    }
    return;
  }

  // 3. delta and wavelength at the pre-step energy and material
  const G4double energy = pre->GetTotalEnergy();
  const G4double delta = 1.0 - GetRefractiveIndex(pre->GetMaterial(), energy);
  const G4double lambda = CLHEP::h_Planck * CLHEP::c_light / energy;
  const G4double length = step->GetStepLength();

  if (!crosses) {
    // whole step
    data.fPhase += -CLHEP::twopi * delta * length / lambda;
    return;
  }

  // fraction of the step up to the plane
  const G4double dz = p1.z() - p0.z();
  const G4double t = dz != 0 ? std::clamp((fPlaneZ - p0.z()) / dz, 0.0, 1.0)
                             : 1.0;
  data.fPhase += -CLHEP::twopi * delta * (t * length) / lambda;
  data.fCounted = true;

  // pixel hit on the plane; the corner of the grid is the lower one
  const G4ThreeVector p = p0 + t * (p1 - p0);
  const G4double x0 = fCenterX - fSizeX * fSpacingX / 2;
  const G4double y0 = fCenterY - fSizeY * fSpacingY / 2;
  const G4double fi = std::floor((p.x() - x0) / fSpacingX);
  const G4double fj = std::floor((p.y() - y0) / fSpacingY);
  const G4bool inside = fi >= 0 && fi < fSizeX && fj >= 0 && fj < fSizeY;

  if (inside) {
    const auto index = static_cast<std::size_t>(fj) * fSizeX +
                       static_cast<std::size_t>(fi);
    const G4double e = energy / CLHEP::keV;
    const G4double amplitude = std::sqrt(e);
    const G4double re = amplitude * std::cos(data.fPhase);
    const G4double im = amplitude * std::sin(data.fPhase);
    G4AutoLock mutex(&WavefrontGridMutex);
    fRealGrid[index] += re;
    fImagGrid[index] += im;
    fEnergyGrid[index] += e;
    fNumberOfCounted++;
    fSumEnergy += e;
  } else {
    G4AutoLock mutex(&WavefrontGridMutex);
    fNumberOutsideGrid++;
  }

  // the first crossing is done, inside or outside the grid
  if (fKillAfterPlane) {
    track->SetTrackStatus(fStopAndKill);
  }
}

py::dict GateWavefrontActor::GetCounters() const {
  G4AutoLock mutex(&WavefrontGridMutex);
  py::dict d;
  d["n_photons_counted"] = fNumberOfCounted;
  d["n_outside_grid"] = fNumberOutsideGrid;
  d["n_recrossings"] = fNumberOfRecrossings;
  d["sum_energy_keV"] = fSumEnergy;
  return d;
}
