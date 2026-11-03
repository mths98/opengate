/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include "GateXRayRefractionProcess.h"

#include <G4Gamma.hh>
#include <G4GeometryTolerance.hh>
#include <G4Material.hh>
#include <G4MaterialPropertiesTable.hh>
#include <G4Navigator.hh>
#include <G4PhysicalConstants.hh>
#include <G4Step.hh>
#include <G4TransportationManager.hh>
#include <G4Track.hh>
#include <Randomize.hh>

#include <cfloat>
#include <cmath>

namespace {
// maximum number of roughness samplings before falling back to the
// geometrical normal
constexpr int kMaxRoughnessTrials = 10;
} // namespace

GateXRayRefractionProcess::GateXRayRefractionProcess(G4double roughnessSigma,
                                                     const G4String &name)
    : G4VDiscreteProcess(name, fOptical), fRoughnessSigma(roughnessSigma) {
  if (fRoughnessSigma < 0) {
    G4ExceptionDescription ed;
    ed << "The surface roughness sigma must be >= 0 rad, got "
       << fRoughnessSigma << " rad.";
    G4Exception("GateXRayRefractionProcess", "XRayRefraction01",
                FatalErrorInArgument, ed);
  }
  fHalfCarTolerance =
      0.5 * G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
}

G4bool
GateXRayRefractionProcess::IsApplicable(const G4ParticleDefinition &particle) {
  return &particle == G4Gamma::Definition();
}

G4double GateXRayRefractionProcess::GetMeanFreePath(const G4Track &, G4double,
                                                    G4ForceCondition *condition) {
  // PostStepDoIt is called at every step; it only acts on boundaries
  *condition = Forced;
  return DBL_MAX;
}

G4double
GateXRayRefractionProcess::GetRefractiveIndex(const G4Material *material,
                                              G4double energy) const {
  G4MaterialPropertyVector *rindex = nullptr;
  auto *mpt = material->GetMaterialPropertiesTable();
  if (mpt != nullptr) {
    rindex = mpt->GetProperty("RINDEX");
  }
  if (rindex == nullptr) {
    G4ExceptionDescription ed;
    ed << "The material '" << material->GetName()
       << "' has no RINDEX property. The x-ray refraction process needs the "
          "refractive index of every material crossed by the photons: it "
          "should have been filled by the xray_phase_contrast option of the "
          "physics manager.";
    G4Exception("GateXRayRefractionProcess", "XRayRefraction02",
                FatalException, ed);
    return 1.0;
  }
  return rindex->Value(energy);
}

G4ThreeVector
GateXRayRefractionProcess::TiltNormal(const G4ThreeVector &normal) const {
  // orthonormal basis (u, v) of the plane tangent to the surface
  const G4ThreeVector u = normal.orthogonal().unit();
  const G4ThreeVector v = normal.cross(u);
  const G4double theta = G4RandGauss::shoot(0.0, fRoughnessSigma);
  const G4double phi = twopi * G4UniformRand();
  return (std::cos(theta) * normal +
          std::sin(theta) * (std::cos(phi) * u + std::sin(phi) * v))
      .unit();
}

G4VParticleChange *GateXRayRefractionProcess::PostStepDoIt(const G4Track &track,
                                                          const G4Step &step) {
  aParticleChange.Initialize(track);

  // 1. only acts on geometrical boundaries, and not on tiny steps
  const auto *postStepPoint = step.GetPostStepPoint();
  if (postStepPoint->GetStepStatus() != fGeomBoundary ||
      step.GetStepLength() <= fHalfCarTolerance) {
    return G4VDiscreteProcess::PostStepDoIt(track, step);
  }

  // 2. refractive indices on both sides of the boundary
  const auto *material1 = step.GetPreStepPoint()->GetMaterial();
  const auto *material2 = postStepPoint->GetMaterial();
  if (material1 == nullptr || material2 == nullptr) {
    return G4VDiscreteProcess::PostStepDoIt(track, step);
  }
  const G4double energy = track.GetTotalEnergy();
  const G4double n1 = GetRefractiveIndex(material1, energy);
  const G4double n2 = GetRefractiveIndex(material2, energy);

  // 3. same refractive index (e.g. same material): no refraction
  if (n1 == n2) {
    return G4VDiscreteProcess::PostStepDoIt(track, step);
  }

  // 4. normal to the boundary, in the global frame, from the tracking
  // navigator
  G4bool valid = false;
  auto *navigator = G4TransportationManager::GetTransportationManager()
                        ->GetNavigatorForTracking();
  G4ThreeVector normal =
      navigator->GetGlobalExitNormal(postStepPoint->GetPosition(), &valid);
  if (!valid) {
    if (!fInvalidNormalWarned) {
      fInvalidNormalWarned = true;
      G4ExceptionDescription ed;
      ed << "The navigator returned an invalid surface normal at "
         << postStepPoint->GetPosition() << " between the materials '"
         << material1->GetName() << "' and '" << material2->GetName()
         << "'. The photon is not refracted. This warning is only emitted "
            "once per thread.";
      G4Exception("GateXRayRefractionProcess", "XRayRefraction03", JustWarning,
                  ed);
    }
    return G4VDiscreteProcess::PostStepDoIt(track, step);
  }

  // 6. orientation: the normal points against the incident direction. It is
  // applied to the geometrical normal, before the roughness, so that a tilt
  // beyond the incident direction can be detected (c1 <= 0) and resampled.
  const G4ThreeVector p = track.GetMomentumDirection();
  normal = normal.unit();
  if (p.dot(normal) > 0) {
    normal = -normal;
  }

  // 5. surface roughness, applied once; resampled until c1 > 0
  if (fRoughnessSigma > 0) {
    for (int i = 0; i < kMaxRoughnessTrials; i++) {
      const G4ThreeVector tilted = TiltNormal(normal);
      if (-p.dot(tilted) > 0) {
        normal = tilted;
        break;
      }
    }
  }

  // 7. vector form of Snell's law
  const G4double r = n1 / n2;
  const G4double c1 = -p.dot(normal);
  const G4double k = 1.0 - r * r * (1.0 - c1 * c1);
  G4ThreeVector newDirection;
  if (k < 0) {
    // total external reflection
    newDirection = p + 2.0 * c1 * normal;
  } else {
    newDirection = r * p + (r * c1 - std::sqrt(k)) * normal;
  }

  // 8. the energy is unchanged
  aParticleChange.ProposeMomentumDirection(newDirection.unit());
  aParticleChange.ProposeVelocity(track.GetVelocity());
  return G4VDiscreteProcess::PostStepDoIt(track, step);
}
