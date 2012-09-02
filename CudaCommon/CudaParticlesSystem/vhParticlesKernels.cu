#include <cutil_math.h>
#include "perlinKernelParts.cu";

__device__ void dampVelDevice(float3* vel, float damping, float dt){

	*vel = *vel * (1 - damping*dt);

}

__global__ void dampVelLeadsKernel(float3* vel, float damping, float dt, int nLeadParts, int trailLength)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nLeadParts) {
		n = n*trailLength;
		vel[n] = (1 - damping*dt)*vel[n];
	}
}

__global__ void dampVelTrailsKernel(float3* vel, float damping, float dt, int nParts, int trailLength)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nParts) {

		if (n%trailLength != 0) {
			vel[n] = (1 - damping*dt)*vel[n];
		}

	}
}

__device__ void addGravityDevice(float3* vel, float3 gravityDir, float gravityStrength, float dt){

	*vel = *vel + gravityDir*gravityStrength*dt;
}

__global__ void addGravityLeadsKernel(float3* vel, float3 gravityDir, float gravityStrength, float dt, int nLeadParts, int trailLength)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nLeadParts) {
		n = n*trailLength;
		addGravityDevice(&vel[n], gravityDir, gravityStrength, dt);
	}
}

__global__ void addGravityTrailsKernel(float3* vel, float3 gravityDir, float gravityStrength, float dt, int nParts, int trailLength)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nParts) {

		 if (n%trailLength != 0) {
			addGravityDevice(&vel[n], gravityDir, gravityStrength, dt);
		}
	}
}

__device__ void addAttractorDevice(float3* vel, float3* pos, float strength, float3 origin, float radius, int decay, float dt){

	float3 dir = *pos - origin;
	float sqrtdist = dot(dir,dir);

	float force = 0;

	if(radius != 0 && sqrtdist > radius*radius){
	} else if (decay == 0 || sqrtdist < 1) {
		force = strength;
	} else if (decay > 0 ) {
			force = strength/sqrtdist;
	}

	force = strength/sqrtdist;

	*vel = *vel + force*dt*dir;
}

__global__ void addAttractorLeadsKernel(float3* vel, float3* pos, float strength, float3 origin, float radius, int decay, float dt, int nLeadParts, int trailLength)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nLeadParts) {
		n = n*trailLength;
		addAttractorDevice(&vel[n], &pos[n], strength, origin, radius, decay, dt);
	}
}

__global__ void addAttractorTrailsKernel(float3* vel, float3* pos, float strength, float3 origin, float radius, int decay, float dt, int nParts, int trailLength)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nParts) {

		 if (n%trailLength != 0) {
			addAttractorDevice(&vel[n], &pos[n], strength, origin, radius, decay, dt);
		}
	}
}

__device__ void addTurbulenceDevice(float3* vel, float3* pos,
								  float3 noiseAmp, float3 noiseOffset, int noiseOct, float noiseLac, float noiseFreq,
								  float dt){

	
	float3 noise = make_float3(0,0,0);
	if (noiseAmp.x != 0)
		noise.x = noiseAmp.x*noise1D((*pos).x+noiseOffset.x, (*pos).y+noiseOffset.y, (*pos).z+noiseOffset.z,
									noiseOct, noiseLac, 0.5, noiseFreq,1);
	if (noiseAmp.y != 0)
		noise.y  = noiseAmp.y*noise1D((*pos).x+noiseOffset.x+2000, (*pos).y+noiseOffset.y, (*pos).z+noiseOffset.z,
									noiseOct, noiseLac, 0.5, noiseFreq,1);
	if (noiseAmp.x != 0)
		noise.z  += noiseAmp.z*noise1D((*pos).x+noiseOffset.x+5000, (*pos).y+noiseOffset.y, (*pos).z+noiseOffset.z,
									noiseOct, noiseLac, 0.5, noiseFreq,1);

	*vel = *vel + noise*dt;

}

__global__ void addTurbulenceLeadsKernel(float3* vel, float3* pos,
								  float3 noiseAmp, float3 noiseOffset, int noiseOct, float noiseLac, float noiseFreq,
								  float dt, int nLeadParts, int trailLength)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nLeadParts) {
		n = n*trailLength;
		addTurbulenceDevice(&vel[n], &pos[n], noiseAmp, noiseOffset, noiseOct, noiseLac, noiseFreq, dt);

	}
}

__global__ void addTurbulenceTrailsKernel(float3* vel, float3* pos,
								  float3 noiseAmp, float3 noiseOffset, int noiseOct, float noiseLac, float noiseFreq,
								  float dt, int nParts, int trailLength)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nParts) {

		if (n%trailLength != 0) {
			addTurbulenceDevice(&vel[n], &pos[n], noiseAmp, noiseOffset, noiseOct, noiseLac, noiseFreq, dt);
		}

	}
}


__global__ void emitTrailsReordKernel(float3* pos, float3* vel, float* age, float* life, char* opafix,
									  float inheritVel, float inheritAge, float partsLife,
									  float dt, int nLeadParts, int trailLength)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nLeadParts) {

		n = n*trailLength;

		int t = trailLength+1;

		if (age[n]>=0) {
			t = rintf(age[n]/dt);

			if( t<trailLength) {
				age[n+t] = 0.0;
				life[n+t] = partsLife;
			}

			for (int i = trailLength-1; i>1; i--) {

				pos[n+i] = pos[n+i-1];
				vel[n+i] = vel[n+i-1];

				if(age[n+i-1]>0)
					age[n+i] = age[n+i-1];


				if(i!= (trailLength-1)){
					if(age[n+i+1]<0 /*|| age[n+i-1]<0*/)
						opafix[n+i] = 1;
					else
						opafix[n+i] = 0;
				}
			}

			pos[n+1] = pos[n];
			vel[n+1] = inheritVel*vel[n];
			age[n+1] = inheritAge*age[n];
			opafix[n+1] = 0;
		} else {
			opafix[n+1] = 1;
		}

	}
}

__global__ void emitTrailsKernel(float3* pos, float3* vel, float* age, float* life,
								 float inheritVel, float inheritAge, float partsLife,
								 float dt, int nLeadParts, int trailLength)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nLeadParts) {

		n = n*trailLength;

		int t = trailLength+1;

		if (age[n]>0) {
			t = rintf(age[n]/dt);

			t = t%trailLength;

			if (t!=0) {
				pos[n+t] = pos[n];
				vel[n+t] = inheritVel * vel[n];
				age[n+t] = inheritAge * age[n];
				life[n+t] = partsLife;
			}
		}

	}
}


__global__ void integrateParticlesKernel(float3* pos, float3* vel, float* age, float* life, float partsLife,
								  float4* colour, float opacity, float3 col1, float3 col2, char* opafix, float dt, int nParts,int trailLength)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nParts) {

		if (age[n]>=0) {

			pos[n] = pos[n] + vel[n]*dt;

			age[n] = age[n] + dt;

			if (age[n] > life[n]) {
				age[n] = -1.0;
				life[n] = -1.0;
			}

		}

		float ageNorm = age[n]/(life[n]);

		float3 col = lerp(col1,col2,ageNorm);
		float alpha = opacity*(1-pow(ageNorm,2));

		if (age[n]<0 || opafix[n] == 1)
			alpha = 0.0;

		/*if (opafix[n])
			alpha = 0.0;*/
	
		colour[n] = make_float4(col.x,col.y,col.z,alpha);
		

	}
}


__global__ void initNewParticlesKernel(float3* pos, float3* vel, float* age, float* life,
										float3 initPos, float3 initVel, float radVelAmp,
										float3 noiseVelAmp, float3 noiseVelOffset, int noiseVelOct, float noiseVelLac, float noiseVelFreq,
										float initLife, float time, int nNewLeadParts, int trailLength)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	//__device__ inline float noise1D(float x, float y, float z, int octaves,
	//	     float lacunarity, float gain, float freq, float amp)

	if (n<nNewLeadParts) {

		n = n*trailLength;
						
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

		for (int i = 1; i<trailLength; i++){
			pos[n+i] = pos[n];
			//age[n+i] = -1.0;
		}


	}

	
}

	

__global__ void resetParticlesKernel(float3* pos, float3* vel, float* age, float* life, char* opafix, int nParts)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nParts) {
		pos[n] = make_float3(2.0,0.0,0.0);
		vel[n] =  make_float3(0.0,0.0,0.0);
		age[n] = -1.0;
		life[n] = -1.0;
		opafix[n] = 0;
	}
}

__global__ void calcDepthKernel(float3* pos, float* keys, unsigned int *indices, float3 vector, int nParts)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n < nParts) {

		keys[n] = -dot(pos[n], vector);        // project onto sort vector

		indices[n] = n;

	}
}


extern "C" void dampVelCu(float3* vel, float damping, float dt, int nLeadParts, int trailLength, int leads){

	if (leads){

		int nthreads = min(256, nLeadParts);
		int nBlocks = nLeadParts/nthreads + (!(nLeadParts%nthreads)?0:1);
		dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

		dampVelLeadsKernel<<< blocks, threads>>>(vel, damping, dt, nLeadParts, trailLength);

	} else {

		int nParts = nLeadParts * trailLength;
		int nthreads = min(256, nParts);
		int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);
		dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

		dampVelTrailsKernel<<< blocks, threads>>>(vel, damping, dt, nParts, trailLength);
	}
}

extern "C" void addGravityCu(float3* vel, float3 gravityDir, float gravityStrength, float dt, int nLeadParts, int trailLength, int leads){

	if (leads) {
		int nthreads = min(256, nLeadParts);
		int nBlocks = nLeadParts/nthreads + (!(nLeadParts%nthreads)?0:1);
		dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

		addGravityLeadsKernel<<< blocks, threads>>>(vel, gravityDir, gravityStrength, dt, nLeadParts, trailLength);

	} else {
		int nParts = nLeadParts * trailLength;
		int nthreads = min(256, nParts);
		int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);
		dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

		addGravityTrailsKernel<<< blocks, threads>>>(vel, gravityDir, gravityStrength, dt, nParts, trailLength);
	}
}

extern "C" void addAttractorCu(float3* vel, float3* pos, float strength, float3 origin, float radius, int decay, float dt, int nLeadParts, int trailLength, int leads){

	if (leads) {
		int nthreads = min(256, nLeadParts);
		int nBlocks = nLeadParts/nthreads + (!(nLeadParts%nthreads)?0:1);
		dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

		addAttractorLeadsKernel<<< blocks, threads>>>(vel, pos, strength, origin, radius, decay, dt, nLeadParts, trailLength);

	} else {
		int nParts = nLeadParts * trailLength;
		int nthreads = min(256, nParts);
		int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);
		dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

		addAttractorTrailsKernel<<< blocks, threads>>>(vel, pos, strength, origin, radius, decay, dt, nParts, trailLength);
	}
}

extern "C" void addTurbulenceCu(float3* vel, float3* pos, float3 noiseAmp, float3 noiseOffset,
								int noiseOct, float noiseLac, float noiseFreq, float dt, int nLeadParts, int trailLength, int leads){

	if (leads) {

		int nthreads = min(256, nLeadParts);
		int nBlocks = nLeadParts/nthreads + (!(nLeadParts%nthreads)?0:1);
		dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

		addTurbulenceLeadsKernel<<< blocks, threads>>>(vel, pos, noiseAmp, noiseOffset, noiseOct, noiseLac, noiseFreq, dt, nLeadParts, trailLength);
	} else {

		int nParts = nLeadParts * trailLength;
		int nthreads = min(256, nParts);
		int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);
		dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

		addTurbulenceTrailsKernel<<< blocks, threads>>>(vel, pos, noiseAmp, noiseOffset, noiseOct, noiseLac, noiseFreq, dt, nParts, trailLength);
	}
}

extern "C" void integrateParticlesCu(float3* pos, float3* vel, float* age, float* life, float partsLife,
								  float4* colour, float opacity, float3 col1, float3 col2, char* opafix,
									float dt, int nParts, int trailLength){

	int nthreads = min(256, nParts);
	int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);
    dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

	integrateParticlesKernel<<< blocks, threads>>>(pos, vel, age, life, partsLife, colour, opacity, col1, col2, opafix, dt, nParts, trailLength);
}

extern "C" void emitTrailsCu(float3* pos, float3* vel, float* age, float* life, char* opafix,
							 float inheritVel, float inheritAge, float partsLife,
							 float dt, int nLeadParts, int trailLength, int reorder){

	int nthreads = min(256, nLeadParts);
	int nBlocks = nLeadParts/nthreads + (!(nLeadParts%nthreads)?0:1);
    dim3 blocks(nBlocks, 1,1); dim3 threads(nthreads, 1, 1);

	if (reorder)
		emitTrailsReordKernel<<< blocks, threads>>>(pos, vel, age, life,  opafix, inheritVel, inheritAge, partsLife,
													dt, nLeadParts, trailLength);
	else
		emitTrailsKernel<<< blocks, threads>>>(pos, vel, age, life, inheritVel, inheritAge, partsLife,
												dt, nLeadParts, trailLength);
}

extern "C" void resetParticlesCu(float3* pos, float3* vel, float* age, float* life, char* opafix, int nParts) {

	int nthreads = min(256, nParts);

	int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);

    dim3 blocks(nBlocks, 1,1);
    dim3 threads(nthreads, 1, 1);

    resetParticlesKernel<<< blocks, threads>>>(pos, vel, age, life, opafix, nParts);
}

extern "C" void initNewParticlesCu(float3* pos, float3* vel, float* age, float* life,
									float3 initPos, float3 initVel, float radVelAmp, 
									float3 noiseVelAmp, float3 noiseVelOffset, int noiseVelOct, float noiseVelLac, float noiseVelFreq,
									float initLife, float time, int nParts, int trailLength) {

	int nthreads = min(256, nParts);

	int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);

    dim3 blocks(nBlocks, 1,1);
    dim3 threads(nthreads, 1, 1);

	cudaMemcpyToSymbol(c_perm_3d, h_perm, sizeof(h_perm),0,cudaMemcpyHostToDevice );

    initNewParticlesKernel<<< blocks, threads>>>(pos, vel, age, life, initPos, initVel, radVelAmp,
												noiseVelAmp, noiseVelOffset, noiseVelOct, noiseVelLac, noiseVelFreq,
												initLife, time, nParts, trailLength);

}

extern "C" void calcDepthCu(float3* pos, float* keys, unsigned int *indices, float3 vector, int nParts) {

	int nthreads = min(256, nParts);

	int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);

    dim3 blocks(nBlocks, 1,1);
    dim3 threads(nthreads, 1, 1);

	calcDepthKernel<<<blocks, threads>>>(pos, keys, indices, vector, nParts);

}