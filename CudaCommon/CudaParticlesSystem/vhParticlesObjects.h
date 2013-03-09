#ifndef __VHPARTICLESOBJECTS_H__
#define __VHPARTICLESOBJECTS_H__

#include <windows.h> //very important
#include <cmath>
//#include <stdio.h> //very important

namespace cu{

	#include <vector_types.h>

}

#include "../CudaFluidSolver3D/vhFluidSolver3D.h"

struct VHParticlesSystem;

struct ParticlesEmitter {

	cu::float3 pos;

	float radius;
	float amount;

	cu::float3 vel;

	float radVelAmp;

	cu::float3 noiseVelAmp;

	cu::float3 noiseVelOffset;

	int noiseVelOct;
	float noiseVelLac;
	float noiseVelFreq;

	
};

struct ParticlesCollider {

	cu::float3 pos;
	cu::float3 oldPos;

	float radius;
	
};

/*struct ParticlesLight {

	vec3f				m_lightPos, m_lightTarget, m_lightColor, m_colorAttenuation;
	float shadowAlpha;


};*/

struct ParticlesForce {

	ParticlesForce();
	ParticlesForce(VHParticlesSystem* newSys);
	~ParticlesForce();

	VHParticlesSystem* pSys;
	float strength;

	virtual void applyForce(int lead)=0;


};

struct DampingForce : public ParticlesForce {

	DampingForce(VHParticlesSystem* newSys);

	void applyForce(int lead);

};

struct GravityForce : public ParticlesForce {

	GravityForce(VHParticlesSystem* newSys);

	cu::float3 gravityDir;

	void applyForce(int lead);

};

struct TurbulenceForce : public ParticlesForce {

	TurbulenceForce(VHParticlesSystem* newSys);

	cu::float3 noiseAmp;
	cu::float3 noiseOffset;
	int noiseOct;
	float noiseLac;
	float noiseFreq;

	void applyForce(int lead);

};


struct FluidForce : public ParticlesForce {

	FluidForce(VHParticlesSystem* newSys);

	VHFluidSolver3D* fluidSolver;

	void applyForce(int lead);

};

struct AttractorForce : public ParticlesForce {

	AttractorForce(VHParticlesSystem* newSys);

	cu::float3 origin;
	float radius;
	int decay;

	void applyForce(int lead);

};

extern "C" void dampVelCu(cu::float3* vel, float damping, float dt, int nLeadParts, int trailLength, int leads);

extern "C" void addGravityCu(cu::float3* vel, cu::float3 gravityDir, float gravityStrength,
							 float dt, int nLeadParts, int trailLength, int leads);

extern "C" void addTurbulenceCu(cu::float3* vel, cu::float3* pos, cu::float3 noiseAmp, cu::float3 noiseOffset,
								int noiseOct, float noiseLac, float noiseFreq, float dt, int nLeadParts, int trailLength, int leads);

extern "C" void addAttractorCu(cu::float3* vel, cu::float3* pos, float strength, cu::float3 origin, float radius,
							   float decay, float dt, int nLeadParts, int trailLength, int leads);

#endif 