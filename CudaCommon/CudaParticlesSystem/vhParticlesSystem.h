#ifndef __VHPARTICLESSYTEM_H__
#define __VHPARTICLESSYTEM_H__

#include <stdlib.h>
#include <stdio.h> //very important
#include <cmath>

#include "vhParticlesRender.h"
#include "../../Utils/cudaVbo.h"


namespace cu{
	#include <cuda_runtime_api.h>
	//#include <cuda_gl_interop.h>
	#include <vector_functions.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>
}

#include "vhParticlesObjects.h"


struct VHParticlesSystem {

	VHParticlesSystem();
	~VHParticlesSystem();

	static VHParticlesSystem* systemsList[10];
	static int numSystems;

	int id;

	VHParticlesRender* pRend;


	int preview;
	float opacity;
	cu::float3 startColor;
	cu::float3 endColor;

	//------------------

	float time;
	float dt;

	float partsLife;
	float partsLifeVar;

	float inheritVel;
	float inheritAge;


	ParticlesForce* leadsForces[10];
	ParticlesForce* trailsForces[10];
	int nLeadsForces;
	int nTrailsForces;

	int nLeadParts;
	int trailLength;
	int nParts;
	int index;

	int nEmit;
	ParticlesEmitter* emitters;

	CudaVbo<cu::float3>		*posVbo;
	CudaVbo<cu::float4>		*colourVbo;
	CudaVbo<unsigned int>	*indexVbo;

	float			*host_pos;
	float			*host_vel;

	cu::float3		*dev_pos;
	cu::float3      *dev_vel;
	cu::float4      *dev_colour;
	float			*dev_age;
	float			*dev_life;
	char			*dev_opafix;

	unsigned int	*dev_indices;
	float			*dev_keys;
	

	cu::float3		*tempdev_pos;
	cu::float3		*tempdev_vel;
	float		*tempdev_age;
	float		*tempdev_life;

	void initParticlesSystem(int newLeadParts, int newTrailLength);
	void freeParticlesSystem();
	void changeMaxParts(int newLeadParts, int newTrailLength);
	void resetParticles();
	void emitParticles();
	void updateParticles();

	void updateVBOs();
	void draw();

	virtual void lockOpenGLContext();
	virtual void unlockOpenGLContext();

};

extern "C" void integrateParticlesCu(cu::float3* pos, cu::float3* vel, float* age, float* life, float partsLife,
								  cu::float4* colour, float opacity, cu::float3 col1, cu::float3 col2, char* opafix,
									float dt, int nParts, int trailLength);

extern "C" void emitTrailsCu(cu::float3* pos, cu::float3* vel, float* age, float* life, char* opafix,
							  float inheritVel, float inheritAge, float partsLife,
							 float dt, int nLeadParts, int trailLength, int reorder);


extern "C" void resetParticlesCu(cu::float3* pos, cu::float3* vel, float* age, float* life, char* opafix, int nParts);

extern "C" void initNewParticlesCu(cu::float3* pos, cu::float3* vel, float* age, float* life,
								   cu::float3 initPos, cu::float3 initVel, float radVelAmp,
								   cu::float3 noiseVelmp, cu::float3 noiseVelOffset, int noiseVelOct, float noiseVelLac, float noiseVelFreq,
								   float initLife, float time, int nNewLeadParts, int trailLength);

extern "C" void calcDepthCu(cu::float3* pos, float* keys, unsigned int *indices, cu::float3 vector, int nParts);

#endif  // __DATABLOCK_H__