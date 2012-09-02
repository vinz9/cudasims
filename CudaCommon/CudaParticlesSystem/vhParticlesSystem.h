#ifndef __VHPARTICLESSYTEM_H__
#define __VHPARTICLESSYTEM_H__

#include "vhParticlesObjects3D.h"
#include <stdlib.h>
#include <stdio.h> //very important
#include <cmath>


namespace cu{
	#include <cuda_runtime_api.h>
	#include <cuda_gl_interop.h>
	#include <vector_functions.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>
}

#include "../CudaFluidSolver3D/vhFluidSolver3D.h"

struct VHParticlesSystem {

	VHParticlesSystem();
	~VHParticlesSystem();

	static VHParticlesSystem* systemsList[10];
	static int numSystems;

	int id;

	VHFluidSolver3D* fluidSolver;

	unsigned int g_textureID;

	//GLuint posVbo;
	unsigned int posVbo;
	struct cu::cudaGraphicsResource *posVboRes;

	//GLuint colourVbo;
	unsigned int colourVbo;
	struct cu::cudaGraphicsResource *colourVboRes;

	float time;
	float dt;

	float partsLife;
	float partsLifeVar;
	float velDamp;
	float gravityStrength;
	cu::float3 gravityDir;
	float fluidStrength;

	cu::float3 noiseAmp;
	cu::float3 noiseOffset;
	int noiseOct;
	float noiseLac;
	float noiseFreq;

	int preview;
	float opacity;
	float pointSize;
	cu::float3 startColor;
	cu::float3 endColor;


	int nParts;
	int index;

	int nEmit;
	ParticlesEmitter* emitters;


	float			*host_pos;
	float			*host_vel;

	cu::float3		*dev_pos;
	cu::float3      *dev_vel;
	cu::float4      *dev_colour;
	float			*dev_age;
	float			*dev_life;

	cu::float3		*tempdev_pos;
	cu::float3		*tempdev_vel;
	float		*tempdev_age;
	float		*tempdev_life;

	void initParticlesSystem(int maxParts);
	void freeParticlesSystem();
	void changeMaxParts(int maxParts);
	void resetParticles();
	void emitParticles();
	void updateParticles();

	void updateVBO(cu::float3* destPos, cu::float4* destColour);
	void draw();

	virtual void lockOpenGLContext();
	virtual void unlockOpenGLContext();

};

extern "C" void dampVelCu(cu::float3* vel, float damping, float dt, int nParts);

extern "C" void addGravityCu(cu::float3* vel, cu::float3 gravityDir, float gravityStrength, float dt, int nParts);

extern "C" void addTurbulenceCu(cu::float3* vel, cu::float3* pos, cu::float3 noiseAmp, cu::float3 noiseOffset,
								int noiseOct, float noiseLac, float noiseFreq, float dt, int nParts);

extern "C" void integrateParticlesCu(cu::float3* pos, cu::float3* vel, float* age, float* life,
								  cu::float4* colour, float opacity, cu::float3 col1, cu::float3 col2,
									float dt, int nParts);

extern "C" void resetParticlesCu(cu::float3* pos, cu::float3* vel, float* age, float* life, int nParts);

extern "C" void initNewParticlesCu(cu::float3* pos, cu::float3* vel, float* age, float* life,
								   cu::float3 initPos, cu::float3 initVel, float radVelAmp,
								   cu::float3 noiseVelmp, cu::float3 noiseVelOffset, int noiseVelOct, float noiseVelLac, float noiseVelFreq,
								   float initLife, float time, int nParts);

#endif  // __DATABLOCK_H__