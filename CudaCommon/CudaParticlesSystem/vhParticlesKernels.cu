#include <cutil_math.h>
#include "perlinKernelParts.cu";

__global__ void dampVelKernel(float3* vel, float damping, float dt, int nParts)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nParts) {

		vel[n] = (1 - damping*dt)*vel[n];

	}
}

__global__ void addGravityKernel(float3* vel, float3 gravityDir, float gravityStrength, float dt, int nParts)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nParts) {
		vel[n] = vel[n] + gravityDir*gravityStrength*dt;
	}
}

__global__ void addTurbulenceKernel(float3* vel, float3* pos,
								  float3 noiseAmp, float3 noiseOffset, int noiseOct, float noiseLac, float noiseFreq,
								  float dt, int nParts)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nParts) {

		float3 currVel = vel[n];
		
		float3 noise = make_float3(0,0,0);
		if (noiseAmp.x != 0)
			noise.x = noiseAmp.x*noise1D(pos[n].x+noiseOffset.x, pos[n].y+noiseOffset.y, pos[n].z+noiseOffset.z,
										noiseOct, noiseLac, 0.5, noiseFreq,1);
		if (noiseAmp.y != 0)
			noise.y  = noiseAmp.y*noise1D(pos[n].x+noiseOffset.x+2000, pos[n].y+noiseOffset.y, pos[n].z+noiseOffset.z,
										noiseOct, noiseLac, 0.5, noiseFreq,1);
		if (noiseAmp.x != 0)
			noise.z  += noiseAmp.z*noise1D(pos[n].x+noiseOffset.x+5000, pos[n].y+noiseOffset.y, pos[n].z+noiseOffset.z,
										noiseOct, noiseLac, 0.5, noiseFreq,1);

		vel[n] = vel[n] + noise*dt;

	}
}

__global__ void integrateParticlesKernel(float3* pos, float3* vel, float* age, float* life,
								  float4* colour, float opacity, float3 col1, float3 col2, float dt, int nParts)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nParts) {

		pos[n] = pos[n] + vel[n]*dt;

		age[n] = age[n] + dt;
		if (age[n] > life[n])
			age[n] = life[n];

		float ageNorm = age[n]/life[n];

		float3 col = lerp(col1,col2,ageNorm);
		float alpha = opacity*(1-pow(age[n]/life[n],2));

		colour[n] = make_float4(col.x,col.y,col.z,alpha);


	}
}

__global__ void initNewParticlesKernel(float3* pos, float3* vel, float* age, float* life,
										float3 initPos, float3 initVel, float radVelAmp,
										float3 noiseVelAmp, float3 noiseVelOffset, int noiseVelOct, float noiseVelLac, float noiseVelFreq,
										float initLife, float time, int nParts)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	//__device__ inline float noise1D(float x, float y, float z, int octaves,
	//	     float lacunarity, float gain, float freq, float amp)

	if (n<nParts) {
						
		float3 radVel = radVelAmp * (pos[n] - initPos);	
		
		vel[n] = initVel + radVel;
							
		if (noiseVelAmp.x != 0)
			vel[n].x += noiseVelAmp.x*noise1D(pos[n].x+noiseVelOffset.x, pos[n].y+noiseVelOffset.y, pos[n].z+noiseVelOffset.z,
										noiseVelOct, noiseVelLac, 0.5, noiseVelFreq,1);
			
		if (noiseVelAmp.y != 0)
			vel[n].y  += noiseVelAmp.y*noise1D(pos[n].x+noiseVelOffset.x+2000, pos[n].y+noiseVelOffset.y, pos[n].z+noiseVelOffset.z,
										noiseVelOct, noiseVelLac, 0.5, noiseVelFreq,1);
	
		if (noiseVelAmp.x != 0)
			vel[n].z  += noiseVelAmp.z*noise1D(pos[n].x+noiseVelOffset.x+5000, pos[n].y+noiseVelOffset.y, pos[n].z+noiseVelOffset.z,
										noiseVelOct, noiseVelLac, 0.5, noiseVelFreq,1);
	
		age[n] = 0.0;
	}

	
}

__global__ void resetParticlesKernel(float3* pos, float3* vel, float* age, float* life, int nParts)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nParts) {
		pos[n] = make_float3(0.0,0.0,0.0);
		vel[n] =  make_float3(0.0,0.0,0.0);
		age[n] = 1.0;
		life[n] = 1.0;
	}
}


extern "C" void dampVelCu(float3* vel, float damping, float dt, int nParts){

	int nthreads = min(256, nParts);
	int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);
    dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

	dampVelKernel<<< blocks, threads>>>(vel, damping, dt, nParts);
}

extern "C" void addGravityCu(float3* vel, float3 gravityDir, float gravityStrength, float dt, int nParts){

	int nthreads = min(256, nParts);
	int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);
    dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

	addGravityKernel<<< blocks, threads>>>(vel, gravityDir, gravityStrength, dt, nParts);
}

extern "C" void addTurbulenceCu(float3* vel, float3* pos, float3 noiseAmp, float3 noiseOffset,
								int noiseOct, float noiseLac, float noiseFreq, float dt, int nParts){

	int nthreads = min(256, nParts);
	int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);
    dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

	addTurbulenceKernel<<< blocks, threads>>>(vel, pos, noiseAmp, noiseOffset, noiseOct, noiseLac, noiseFreq, dt, nParts);
}

extern "C" void integrateParticlesCu(float3* pos, float3* vel, float* age, float* life,
								  float4* colour, float opacity, float3 col1, float3 col2,
									float dt, int nParts){

	int nthreads = min(256, nParts);
	int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);
    dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

	integrateParticlesKernel<<< blocks, threads>>>(pos, vel, age, life, colour, opacity, col1, col2, dt, nParts);

}


extern "C" void resetParticlesCu(float3* pos, float3* vel, float* age, float* life, int nParts) {

	int nthreads = min(256, nParts);

	int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);

    dim3 blocks(nBlocks, 1,1);
    dim3 threads(nthreads, 1, 1);

    resetParticlesKernel<<< blocks, threads>>>(pos, vel, age, life, nParts);
}

extern "C" void initNewParticlesCu(float3* pos, float3* vel, float* age, float* life,
									float3 initPos, float3 initVel, float radVelAmp, 
									float3 noiseVelAmp, float3 noiseVelOffset, int noiseVelOct, float noiseVelLac, float noiseVelFreq,
									float initLife, float time, int nParts) {

	int nthreads = min(256, nParts);

	int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);

    dim3 blocks(nBlocks, 1,1);
    dim3 threads(nthreads, 1, 1);

	cudaMemcpyToSymbol(c_perm_3d, h_perm, sizeof(h_perm),0,cudaMemcpyHostToDevice );

    initNewParticlesKernel<<< blocks, threads>>>(pos, vel, age, life, initPos, initVel, radVelAmp,
												noiseVelAmp, noiseVelOffset, noiseVelOct, noiseVelLac, noiseVelFreq,
												initLife, time, nParts);
}