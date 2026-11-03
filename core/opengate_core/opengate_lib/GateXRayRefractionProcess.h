/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#ifndef GateXRayRefractionProcess_h
#define GateXRayRefractionProcess_h

#include <G4ThreeVector.hh>
#include <G4VDiscreteProcess.hh>

/*
  Geant4 process refracting x-ray photons (gamma) at the geometrical
  boundaries between materials, following the vector form of Snell's law with
  the real part of the complex refractive index n = 1 - delta + i beta.

  The refractive index is read from the "RINDEX" property of the material
  properties table of the materials on both sides of the boundary, at the
  photon energy. These tables are filled by the x-ray phase contrast option of
  the physics manager (xray_phase_contrast).

  The process is forced: Geant4 calls PostStepDoIt at every step, but the
  direction only changes when the step ends on a geometrical boundary between
  two materials with different refractive indices. The energy is never
  changed. Total external reflection is applied when Snell's law has no real
  solution.

  An optional surface roughness tilts the surface normal once, by an angle
  theta ~ N(0, sigma) in a uniformly distributed azimuth.

  Each worker thread owns its own instance of the process, so the members
  below are not shared between threads.
*/
class GateXRayRefractionProcess : public G4VDiscreteProcess {
public:
  explicit GateXRayRefractionProcess(G4double roughnessSigma = 0.0,
                                     const G4String &name = "XRayRefraction");

  ~GateXRayRefractionProcess() override = default;

  G4bool IsApplicable(const G4ParticleDefinition &particle) override;

  G4double GetMeanFreePath(const G4Track &track, G4double previousStepSize,
                           G4ForceCondition *condition) override;

  G4VParticleChange *PostStepDoIt(const G4Track &track,
                                  const G4Step &step) override;

private:
  // Refractive index (real part) of the material at the given energy. Raises
  // a fatal G4Exception if the material has no RINDEX property.
  G4double GetRefractiveIndex(const G4Material *material,
                              G4double energy) const;

  // Normal tilted by the surface roughness: angle theta ~ N(0, sigma) in a
  // uniformly distributed azimuth around the given (unit) normal.
  G4ThreeVector TiltNormal(const G4ThreeVector &normal) const;

  G4double fRoughnessSigma = 0.0;
  G4double fHalfCarTolerance = 0.0;
  // the invalid normal warning is only emitted once per thread
  G4bool fInvalidNormalWarned = false;
};

#endif // GateXRayRefractionProcess_h
