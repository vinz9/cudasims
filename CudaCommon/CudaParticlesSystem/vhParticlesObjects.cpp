#include "vhParticlesObjects.h"
#include "vhParticlesSystem.h"


ParticlesForce::ParticlesForce() {

}

ParticlesForce::ParticlesForce(VHParticlesSystem* newSys){

	pSys = newSys;
	strength = 0;

}

ParticlesForce::~ParticlesForce() {

}

DampingForce::DampingForce(VHParticlesSystem* newSys) : ParticlesForce(newSys) { }

void DampingForce::applyForce(int lead) {

	if(strength != 0)
		dampVelCu(pSys->dev_vel, strength, pSys->dt, pSys->nLeadParts, pSys->trailLength, lead);

}

GravityForce::GravityForce(VHParticlesSystem* newSys): ParticlesForce(newSys) {

	gravityDir = cu::make_float3(0,-1.0,0);

}

void GravityForce::applyForce(int lead) {

	if(strength != 0)
		addGravityCu(pSys->dev_vel, gravityDir, strength, pSys->dt, 
					pSys->nLeadParts, pSys->trailLength, lead);


}

TurbulenceForce::TurbulenceForce(VHParticlesSystem* newSys): ParticlesForce(newSys) { 

	noiseAmp = cu::make_float3(1,1,1);
	noiseOffset = cu::make_float3(0,0,0);
	noiseOct = 1;
	noiseLac = 2;
	noiseFreq = 1;

}

void TurbulenceForce::applyForce(int lead) {

	if(strength != 0 ) {
		cu::float3 scaledStrength = cu::make_float3(strength*noiseAmp.x, strength*noiseAmp.y, strength*noiseAmp.z);
		addTurbulenceCu(pSys->dev_vel, pSys->dev_pos, scaledStrength, noiseOffset, noiseOct, noiseLac, noiseFreq,
						pSys->dt, pSys->nLeadParts, pSys->trailLength, lead);
	}


}

FluidForce::FluidForce(VHParticlesSystem* newSys): ParticlesForce(newSys) { }

void FluidForce::applyForce(int lead) {

		if(strength != 0 && fluidSolver != NULL) {

			cu::float3 invSize = cu::make_float3(1.0/fluidSolver->fluidSize.x,1.0/fluidSolver->fluidSize.y,
												1.0/fluidSolver->fluidSize.z);

			bindVelTex(fluidSolver->velArray);
			addFluidForceKernelCu(pSys->dev_vel, pSys->dev_pos, fluidSolver->res, invSize, strength, pSys->dt,
									pSys->nLeadParts, pSys->trailLength, lead);
			unbindVelTex();

	}

}

AttractorForce::AttractorForce(VHParticlesSystem* newSys): ParticlesForce(newSys) {

	origin = cu::make_float3(0,0,0);
	radius = 0;
	decay = 0;

}

void AttractorForce::applyForce(int lead) {

		if(strength != 0) {
			addAttractorCu(pSys->dev_vel, pSys->dev_pos, strength, origin, radius, decay, pSys->dt,
									pSys->nLeadParts, pSys->trailLength, lead);

	}

}
