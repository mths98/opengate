/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#ifndef GateXRayRefractionPhysics_h
#define GateXRayRefractionPhysics_h

#include <G4VPhysicsConstructor.hh>

/*
  Geant4 physics constructor adding the x-ray refraction process
  (GateXRayRefractionProcess) to the gamma.

  As for GateUserLimitsPhysics, everything is implemented in C++ on purpose:
  ConstructProcess is called by Geant4 on every worker thread and must not
  re-enter the Python interpreter (see #1163).
*/
class GateXRayRefractionPhysics : public G4VPhysicsConstructor {
public:
  explicit GateXRayRefractionPhysics(G4double roughnessSigma = 0.0);

  ~GateXRayRefractionPhysics() override = default;

  void ConstructParticle() override;

  void ConstructProcess() override;

private:
  G4double fRoughnessSigma = 0.0;
};

#endif // GateXRayRefractionPhysics_h
