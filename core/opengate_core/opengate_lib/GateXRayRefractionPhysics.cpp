/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include "GateXRayRefractionPhysics.h"
#include "GateXRayRefractionProcess.h"

#include <G4ParticleDefinition.hh>
#include <G4ProcessManager.hh>

GateXRayRefractionPhysics::GateXRayRefractionPhysics(G4double roughnessSigma)
    : G4VPhysicsConstructor("XRayRefractionPhysics"),
      fRoughnessSigma(roughnessSigma) {}

void GateXRayRefractionPhysics::ConstructParticle() {}

void GateXRayRefractionPhysics::ConstructProcess() {
  // Called by Geant4 on each worker thread: pure C++, no GIL. The process is
  // created in the calling thread's process table.
  auto *particleIterator = GetParticleIterator();
  particleIterator->reset();
  while ((*particleIterator)()) {
    G4ParticleDefinition *particle = particleIterator->value();
    if (particle->GetParticleName() == "gamma") {
      particle->GetProcessManager()->AddDiscreteProcess(
          new GateXRayRefractionProcess(fRoughnessSigma));
    }
  }
}
