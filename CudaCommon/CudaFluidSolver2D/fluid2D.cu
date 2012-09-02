#include <stdio.h>
#include <cuda.h>
#include <cutil_math.h>

#include "perlinKernel.cu"

#include "vhObjects.h"

#define PI 3.1415926535897932f

texture<float,2>  texNoise;
texture<float2,2>  texVel;
texture<float,2>  texDens;
texture<float,2>  texPressure;
texture<float,2>  texDiv;
texture<float,2>  texVort;
texture<float4,2> texObstacles;


__device__ float linstep2d(float val, float minval, float maxval) {

	return clamp((val-minval)/(maxval-minval), -1.0f, 1.0f);

}

__global__ void obst_to_color( float4 *optr, const float4 *outSrc, int2 gres ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float4 l = outSrc[offset];
		optr[offset].x=optr[offset].y=optr[offset].z=optr[offset].w = l.w;

	}
}

__global__ void float_to_color( float4 *optr, const float *outSrc, int2 gres, float minBound, float maxBound ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float l = outSrc[offset];
		optr[offset].x=optr[offset].y=optr[offset].z=optr[offset].w = linstep2d(l,minBound,maxBound);

	}
}


__global__ void float2_to_color( float4 *optr, const float2 *outSrc, int2 gres,float minBound, float maxBound ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float r = outSrc[offset].x;
		float g = outSrc[offset].y;

		optr[offset].x=linstep2d(r,minBound,maxBound);
		optr[offset].y=linstep2d(g,minBound,maxBound);
		optr[offset].z=0.5;
		optr[offset].w = 1;

	}
}

__global__ void createBorder(float4 *obst, int2 gres, int posX, int negX, int posY, int negY) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
   // int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		obst[offset] = make_float4(0,0,0,0);

		if (negX == 1 && x == 0) {
			obst[offset].w = 1;
		}

		if(posX == 1 && x==(gres.x-1)) {
			obst[offset].w = 1;
		}

		if(negY == 1 && y==0) {
			obst[offset].w = 1;
		}

		if(posY == 1 && y==(gres.y-1))  {
			obst[offset].w = 1;
		}

	}
}

__global__ void addCollider(float4 *obst, float radius, float2 position, int2 gres, float2 vel) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float2 coords = make_float2(x,y);
		float2 pos = (position - coords);
		float scaledRadius = radius;


		if (dot(pos,pos)<(scaledRadius*scaledRadius)) {
			obst[offset].x = vel.x;
			obst[offset].y = vel.y;
			obst[offset].w = 1;
		}

	}
}

__global__ void addDens(float *dens, float timestep, float radius, float amount, float2 position, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
   // int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float2 coords = make_float2(x,y);
		float2 pos = (position - coords);

		if (dot(pos,pos)<(radius*radius))
			dens[offset] += timestep*amount;


	}

}

__global__ void addVel(float2 *vel, float timestep, float radius, float2 strength, float2 position, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float2 coords = make_float2(x,y);
		float2 pos = (position - coords);

		if (dot(pos,pos)<(radius*radius))
			vel[offset] += timestep*strength;

	}
}

__global__ void addDensBuoy(float2 *vel, float timestep, float strength, float2 dir, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
   // int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		vel[offset] += timestep*strength*dir*tex2D(texDens,xc,yc);

	}
}

//Simple kernel fills an array with perlin noise
__global__ void k_perlin(float* noise, unsigned int width, unsigned int height, 
			 float2 delta, unsigned char* d_perm,
			 float time, int octaves, float lacun, float gain, float freq, float amp)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float xCur = ((float) (idx%width)) * delta.x;
  float yCur = ((float) (idx/width)) * delta.y;

 if(threadIdx.x < 256)
    // Optimization: this causes bank conflicts
    s_perm[threadIdx.x] = d_perm[threadIdx.x];
  // this synchronization can be important if there are more that 256 threads
  __syncthreads();

  
  // Each thread creates one pixel location in the texture (textel)
  if(idx < width*height) {
    noise[idx] = noise1D(xCur, yCur, time, octaves, lacun, gain, freq, amp);
	//noise[idx] = noise1D(xCur, yCur, z, octaves, 2.f, 0.75f, 0.3, 0.5);
  }
}

__global__ void addNoise(float2 *vel, float timestep, float strength, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float noise = strength*timestep*tex2D(texNoise,xc,yc)*tex2D(texDens,xc,yc);
		
		vel[offset] += make_float2(noise,noise);

	}
}


__global__ void advectVel(float2 *vel, float timestep, float dissipation, float2 invGridSize, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
   // int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float solid = tex2D(texObstacles,xc,yc).w;

		if (solid > 0) {
			vel[offset] = make_float2(0,0);
			return;
		}

		float2 coords = make_float2(xc,yc);

		float2 pos = coords - timestep * invGridSize * tex2D(texVel,xc,yc)*make_float2((float)gres.x,(float)gres.y);

		vel[offset] = (1-dissipation*timestep) * tex2D(texVel, pos.x,pos.y);

	}

}

__global__ void advectDens(float *dens, float timestep, float dissipation, float2 invGridSize, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float solid = tex2D(texObstacles,xc,yc).w;

		if (solid > 0) {
			dens[offset] = 0;
			return;
		}

		float2 coords = make_float2(xc,yc);

		float2 pos = coords - timestep * invGridSize * tex2D(texVel,xc,yc)*make_float2((float)gres.x,(float)gres.y);

		dens[offset] = (1-dissipation*timestep) * tex2D(texDens, pos.x,pos.y);

	}
}


__global__ void divergence(float *div, int2 gres, float2 invCellSize) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
   // int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float2 vL = tex2D(texVel,xc-1,yc);
		float2 vR = tex2D(texVel,xc+1,yc);
		float2 vT = tex2D(texVel,xc,yc+1);
		float2 vB = tex2D(texVel,xc,yc-1);

		//obstacles
		float4 oL = tex2D(texObstacles,xc-1,yc);
		float4 oR = tex2D(texObstacles,xc+1,yc);
		float4 oT = tex2D(texObstacles,xc,yc+1);
		float4 oB = tex2D(texObstacles,xc,yc-1);

		// Use obstacle velocities for solid cells:
		if (oL.w>0) vL = make_float2(oL.x,oL.y);
		if (oR.w>0) vR = make_float2(oR.x,oR.y);
		if (oT.w>0) vT = make_float2(oT.x,oT.y);
		if (oB.w>0) vB = make_float2(oB.x,oB.y);

		div[offset] = 0.5 * (invCellSize.x*(vR.x - vL.x) + invCellSize.y*(vT.y - vB.y));

	}
}

__global__ void vorticity(float *vort, int2 gres, float2 invCellSize) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float2 vL = tex2D(texVel,xc-1,yc);
		float2 vR = tex2D(texVel,xc+1,yc);
		float2 vT = tex2D(texVel,xc,yc+1);
		float2 vB = tex2D(texVel,xc,yc-1);

		//obstacles
		float4 oL = tex2D(texObstacles,xc-1,yc);
		float4 oR = tex2D(texObstacles,xc+1,yc);
		float4 oT = tex2D(texObstacles,xc,yc+1);
		float4 oB = tex2D(texObstacles,xc,yc-1);

		// Use obstacle velocities for solid cells:
		if (oL.w>0) vL = make_float2(oL.x,oL.y);
		if (oR.w>0) vR = make_float2(oR.x,oR.y);
		if (oT.w>0) vT = make_float2(oT.x,oT.y);
		if (oB.w>0) vB = make_float2(oB.x,oB.y);

		vort[offset] = 0.5 * (invCellSize.x*(vR.y - vL.y) - invCellSize.y*(vT.x - vB.x));

	}
}

__global__ void vortConf(float2 *vel, float timestep, float strength, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float vortL = tex2D(texVort,xc-1,yc);
		float vortR = tex2D(texVort,xc+1,yc);
		float vortT = tex2D(texVort,xc,yc+1);
		float vortB = tex2D(texVort,xc,yc-1);

		float vortC = tex2D(texVort,xc,yc);

		float2 force = 0.5*make_float2(gres.x*(abs(vortT)-abs(vortB)), gres.y*(abs(vortR) - abs(vortL)));
		const float EPSILON = 2.4414e-4; // 2^-12

		float magSqr = max(EPSILON, dot(force, force)); 
  		force *= pow((float)magSqr,(float)-0.5); 
 		force =  strength * vortC * make_float2(1, -1) * force;

		vel[offset] += force*timestep;

	}
}

__global__ void jacobi(float *pressure, float alpha, float rBeta, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float pL = tex2D(texPressure,xc-1,yc);
		float pR = tex2D(texPressure,xc+1,yc);
		float pT = tex2D(texPressure,xc,yc+1);
		float pB = tex2D(texPressure,xc,yc-1);

		float pC = tex2D(texPressure,xc,yc);

		//obstacles
		float4 oL = tex2D(texObstacles,xc-1,yc);
		float4 oR = tex2D(texObstacles,xc+1,yc);
		float4 oT = tex2D(texObstacles,xc,yc+1);
		float4 oB = tex2D(texObstacles,xc,yc-1);

		// Use center pressure for solid cells:
		if (oL.w>0) pL = pC;
		if (oR.w>0) pR = pC;
		if (oT.w>0) pT = pC;
		if (oB.w>0) pB = pC;


		float dC = tex2D(texDiv,xc,yc);

		pressure[offset] = (pL + pR + pB + pT + alpha * dC) * rBeta;

	}
}

__global__ void projection(float2 *vel, int2 gres, float2 invCellSize) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float pL = tex2D(texPressure,xc-1,yc);
		float pR = tex2D(texPressure,xc+1,yc);
		float pT = tex2D(texPressure,xc,yc+1);
		float pB = tex2D(texPressure,xc,yc-1);

		float pC = tex2D(texPressure,xc,yc);

			//obstacles
		float4 oL = tex2D(texObstacles,xc-1,yc);
		float4 oR = tex2D(texObstacles,xc+1,yc);
		float4 oT = tex2D(texObstacles,xc,yc+1);
		float4 oB = tex2D(texObstacles,xc,yc-1);

		float2 obstV = make_float2(0,0);
		float2 vMask = make_float2(1,1);

		if (oT.w > 0) { pT = pC; obstV.y = oT.y; vMask.y = 0; }
		if (oB.w > 0) { pB = pC; obstV.y = oB.y; vMask.y = 0; }
		if (oR.w > 0) { pR = pC; obstV.x = oR.x; vMask.x = 0; }
		if (oL.w > 0) { pL = pC; obstV.x = oL.x; vMask.x = 0; }

		float2 grad = 0.5*make_float2(invCellSize.x*(pR-pL), invCellSize.y*(pT-pB));

		float2 vNew = tex2D(texVel,xc,yc) - grad;

		vel[offset] = vMask*vNew + obstV;

	}
}




static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void swapDataPointers(void **a, void **b) {
	void* temp = *a;
	*a = *b;
	*b = temp;
}

static void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 


void compute2DFluidGridSize(int2 res, dim3 &blocks, dim3 &threads){

	int nThreads = 16;

	threads.x = nThreads;
	threads.y = nThreads;
	threads.z = 1;
	
	blocks.x = res.x/nThreads + (!(res.x%nThreads)?0:1);
	blocks.y = res.y/nThreads + (!(res.y%nThreads)?0:1);
	blocks.z = 1;

}

extern "C" void createBorder2DCu(float4* dev_obstacles, int2 res, int borderPosX, int borderNegX,
																int borderPosY, int borderNegY){

	dim3 threads, blocks;
	compute2DFluidGridSize(res, blocks, threads);
	createBorder<<<blocks,threads>>>(dev_obstacles, res, borderPosX, borderNegX, borderPosY, borderNegY);			

}
																
extern "C" void addCollider2DCu(float4* dev_obstacles, float radius, float2 position, int2 res, float2 vel){

	dim3 threads, blocks;
	compute2DFluidGridSize(res, blocks, threads);	
	addCollider<<<blocks,threads>>>(dev_obstacles, radius, position, res, vel);		

}

extern "C" void advectVel2DCu(float2* dev_vel, float timestep, float velDamp, float2 invGridSize, int2 res){

	dim3 threads, blocks;
	compute2DFluidGridSize(res, blocks, threads);
	advectVel<<<blocks,threads>>>(dev_vel,timestep,velDamp,invGridSize,res);			

}

extern "C" void advectDens2DCu(float* dev_dens, float timestep, float densDis, float2 invGridSize, int2 res){

	dim3 threads, blocks;
	compute2DFluidGridSize(res, blocks, threads);	
	advectDens<<<blocks,threads>>>(dev_dens,timestep,densDis,invGridSize,res);		

}

extern "C" void addDens2DCu(float* dev_dens, float timestep, float radius, float amount,
						float2 position, int2 res){

	dim3 threads, blocks;
	compute2DFluidGridSize(res, blocks, threads);
	addDens<<<blocks,threads>>>(dev_dens,timestep,radius,amount,position,res);			

}

extern "C" void addDensBuoy2DCu(float2* dev_vel, float timestep, float densBuoyStrength,
									float2 densBuoyDir, int2 res){

	dim3 threads, blocks;
	compute2DFluidGridSize(res, blocks, threads);
	addDensBuoy<<<blocks,threads>>>(dev_vel,timestep,densBuoyStrength,densBuoyDir,res);			

}

extern "C" void calcNoise2DCu(float* dev_noise, int2 res, float2 fluidSize, float offset,
							int noiseOct, float noiseLacun, float noiseFreq, float noiseAmp) {

  int nThreads=256; // must be equal or larger than 256! (see s_perm)
  int totalThreads = res.x * res.y;
  int nBlocks = totalThreads/nThreads; 
  nBlocks += ((totalThreads%nThreads)>0)?1:0;

  float xDelta = fluidSize.x/(float)res.x;
  float yDelta = fluidSize.y/(float)res.y;
  
  if(!d_perm) { // for convenience allocate and copy d_perm here
    cudaMalloc((void**) &d_perm,sizeof(h_perm));
    cudaMemcpy(d_perm,h_perm,sizeof(h_perm),cudaMemcpyHostToDevice);
    checkCUDAError("d_perm malloc or copy failed!");
  }

  k_perlin<<< nBlocks, nThreads>>>(dev_noise, res.x, res.y, make_float2(xDelta, yDelta), d_perm, offset,
									noiseOct, noiseLacun, 0.75f, noiseFreq, noiseAmp);
									//3,4,0.75,1,0.5
  
  // make certain the kernel has completed 
  cudaThreadSynchronize();
  checkCUDAError("kernel failed!");


}

extern "C" void addNoise2DCu(float2* dev_vel, float timestep, float noiseStr, int2 res){

	dim3 threads, blocks;
	compute2DFluidGridSize(res, blocks, threads);	
	addNoise<<<blocks,threads>>>(dev_vel, timestep, noiseStr, res);		

}

extern "C" void vorticity2DCu(float* dev_vort, int2 res, float2 invCellSize){

	dim3 threads, blocks;
	compute2DFluidGridSize(res, blocks, threads);
	vorticity<<<blocks,threads>>>(dev_vort,res, invCellSize);			

}

extern "C" void vortConf2DCu(float2* dev_vel, float timestep, float vortConfStrength, int2 res){

	dim3 threads, blocks;
	compute2DFluidGridSize(res, blocks, threads);
	vortConf<<<blocks,threads>>>(dev_vel, timestep, vortConfStrength, res);			

}

extern "C" void divergence2DCu(float* dev_div, int2 res, float2 invCellSize){

	dim3 threads, blocks;
	compute2DFluidGridSize(res, blocks, threads);
	divergence<<<blocks,threads>>>(dev_div,res,invCellSize);			

}

extern "C" void jacobi2DCu(float* dev_pressure, float alpha, float rBeta, int2 res){

	dim3 threads, blocks;
	compute2DFluidGridSize(res, blocks, threads);
	jacobi<<<blocks,threads>>>(dev_pressure, alpha, rBeta, res);			

}

extern "C" void projection2DCu(float2* dev_vel, int2 res, float2 invCellSize){

	dim3 threads, blocks;
	compute2DFluidGridSize(res, blocks, threads);	
	projection<<<blocks,threads>>>(dev_vel,res,invCellSize);		

}

extern "C" void bind2DTexturesCu(cudaArray* noiseArray, cudaArray* velArray, cudaArray* densArray,
				cudaArray* pressureArray, cudaArray* divArray, cudaArray* vortArray, cudaArray* obstArray){
				
	cudaChannelFormatDesc descFloat = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc descFloat2 = cudaCreateChannelDesc<float2>();
	cudaChannelFormatDesc descFloat4 = cudaCreateChannelDesc<float4>();
				
	HANDLE_ERROR(cudaBindTextureToArray(texNoise, noiseArray, descFloat));
	HANDLE_ERROR(cudaBindTextureToArray(texVel, velArray, descFloat2));
	HANDLE_ERROR(cudaBindTextureToArray(texDens, densArray, descFloat));
	HANDLE_ERROR(cudaBindTextureToArray(texPressure, pressureArray, descFloat));
	HANDLE_ERROR(cudaBindTextureToArray(texDiv, divArray, descFloat));
	HANDLE_ERROR(cudaBindTextureToArray(texVort, vortArray, descFloat));
	HANDLE_ERROR(cudaBindTextureToArray(texObstacles, obstArray, descFloat4));
				
				

}

extern "C" void unbind2DTexturesCu(){

	cudaUnbindTexture( texNoise );
	cudaUnbindTexture( texVel );
	cudaUnbindTexture( texDens );
	cudaUnbindTexture( texPressure );
	cudaUnbindTexture( texDiv );
	cudaUnbindTexture( texVort );
	cudaUnbindTexture( texObstacles );

}

extern "C" void setup2DTexturesCu(){

	texDiv.filterMode = cudaFilterModeLinear;
	texPressure.filterMode = cudaFilterModeLinear;
	texDens.filterMode = cudaFilterModeLinear;
	texNoise.filterMode = cudaFilterModeLinear;
	texVel.filterMode = cudaFilterModeLinear;
	texVort.filterMode = cudaFilterModeLinear;
	texObstacles.filterMode = cudaFilterModeLinear;

}

extern "C" void free_d_permCu() {

		if (d_perm) {
			HANDLE_ERROR( cudaFree( d_perm ) );
			d_perm = NULL;
		}
}


extern "C" void render2DFluidCu(float4* d_output, int previewType, float maxBounds, int2 res,
							 float* dev_dens, float2* dev_vel, float* dev_noise, float* dev_pressure,
							 float* dev_vort, float4* dev_obstacles) {

	dim3 threads, blocks;
	compute2DFluidGridSize(res, blocks, threads);	

	if(previewType == 0) {
		float_to_color<<<blocks,threads>>>(d_output, dev_dens,res,0,maxBounds );
	} else if(previewType == 1) {
		float2_to_color<<<blocks,threads>>>(d_output, dev_vel,res,-maxBounds,maxBounds );
	} else if(previewType == 2) {
		float_to_color<<<blocks,threads>>>(d_output, dev_noise,res,-maxBounds,maxBounds );
	} else if(previewType == 3) {
		float_to_color<<<blocks,threads>>>(d_output, dev_pressure,res,-maxBounds,maxBounds );
	} else if(previewType == 4) {
		float_to_color<<<blocks,threads>>>(d_output, dev_vort,res,-maxBounds,maxBounds );
	} else if(previewType == 5) {
		obst_to_color<<<blocks,threads>>>(d_output, dev_obstacles,res );
	}
	

}