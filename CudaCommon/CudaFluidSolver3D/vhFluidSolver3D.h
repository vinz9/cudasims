#ifndef __FLUID3D_H__
#define __FLUID3D_H__

#include "vhFluidObjects3D.h"
#include <stdio.h> //very important

namespace cu{

	#include <cuda_runtime_api.h>
	#include <cuda_gl_interop.h>
	#include <vector_functions.h>

	#include <driver_functions.h>
	#include <channel_descriptor.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>

}

typedef unsigned int  uint;
typedef unsigned char uchar;

struct VHFluidSolver3D {

	VHFluidSolver3D();
	~VHFluidSolver3D();

	static VHFluidSolver3D* solverList[10];
	static int numSolvers;

	int displayX, displayY;
	int displaySliceX, displaySliceY;
	int displayEnum;

	uint gl_Tex;
	uint gl_SliceTex;

	uint pbo;     // OpenGL pixel buffer object
	struct cu::cudaGraphicsResource *cuda_pbo_resource;

	void initPixelBuffer(bool initpbo);

	int id;

	int preview;
	int drawCube;
	float opaScale;
	float stepMul;
	int displayRes;

	int doShadows;
	float shadowDens;
	float shadowStepMul;
	float shadowThres;

	int displaySlice;
	int sliceType;
	int sliceAxis;
	float slicePos;
	float sliceBounds;

	int f;
	int nEmit;
	VHFluidEmitter* emitters;

	int nColliders;
	VHFluidCollider* colliders;

	int fps;
	int substeps;
	int jacIter;
	
	cu::cudaExtent res;

	cu::float3 fluidSize;

	int borderNegX;
	int borderPosX;
	int borderNegY;
	int borderPosY;
	int borderNegZ;
	int borderPosZ;

	float densDis;
	float densBuoyStrength;
	cu::float3 densBuoyDir;

	float velDamp;
	float vortConf;

	float noiseStr;
	float noiseFreq;
	int noiseOct;
	float noiseLacun;
	float noiseSpeed;
	float noiseAmp;

	cu::float3 lightPos;

	int colOutput;

	float			*host_dens;
	float			*host_vel;

	cu::float4		*output_display;
	cu::float4		*output_display_slice;

	float			*dev_noise;
	cu::float4      *dev_vel;
	float			*dev_dens;
	float           *dev_pressure;
	float           *dev_div;
	cu::float4           *dev_vort;
	cu::float4		*dev_obstacles;

	cu::cudaArray *densArray;
	cu::cudaArray *noiseArray;
	cu::cudaArray *velArray;
	cu::cudaArray *divArray;
	cu::cudaArray *pressureArray;
	cu::cudaArray *obstaclesArray;
	cu::cudaArray *vortArray;

    float           totalTime;
    float           frames;

	void changeFluidRes(int x, int y, int z);
	void initFluidSolver(int x, int y, int z);
	void solveFluid();
	void resetFluid();
	void clearFluid();

	void fieldCopy(void* field, cu::cudaArray* fieldArray, size_t size);

	void renderFluid(cu::float4* d_output, double focalLength);
	void renderFluidSlice();

	void drawFluid(float fluidRotX, float fluidRotY, float fluidRotZ,
					float fluidPosX, float fluidPosY, float fluidPosZ);
	void drawFluidSlice(float fluidRotX, float fluidRotY, float fluidRotZ,
					float fluidPosX, float fluidPosY, float fluidPosZ);


	long domainSize( void ) const { return res.width * res.height * res.depth * sizeof(float); }

	virtual void lockOpenGLContext();
	virtual void unlockOpenGLContext();
	virtual void calculateTransRot(float* modelviewH, cu::float3* trans, cu::float3* rot);

};

extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
extern "C" void renderFluidCu(cu::float4 *d_output, uint imageW, uint imageH, float opaScale, cu::cudaExtent res,
							  float focalLength, cu::float3 fluidSize, cu::float3 lightPos,
							float stepMul, float shadowStepMul, float shadowThres, float shadowDens, int doShadows);

extern "C" void renderFluidSliceCu(cu::float4* d_output, cu::cudaExtent res, float slicePos, int sliceAxis, int sliceType,
								   float sliceBounds, float* dev_dens, cu::float4* dev_vel, float* dev_noise,
								   float* dev_pressure, cu::float4* dev_vort, cu::float4* dev_obstacles);

extern "C" void setupTextures();

extern "C" void bindDensTex(cu::cudaArray* densArray);
extern "C" void bindNoiseTex(cu::cudaArray* noiseArray);
extern "C" void bindVelTex(cu::cudaArray* velArray);
extern "C" void bindDivTex(cu::cudaArray* divArray);
extern "C" void bindPressureTex(cu::cudaArray* pressureArray);
extern "C" void bindObstaclesTex(cu::cudaArray* obstaclesArray);
extern "C" void bindVortTex(cu::cudaArray* vortArray);

extern "C" void unbindDensTex();
extern "C" void unbindNoiseTex();
extern "C" void unbindVelTex();
extern "C" void unbindDivTex();
extern "C" void unbindPressureTex();
extern "C" void unbindObstaclesTex();
extern "C" void unbindVortTex();

extern "C" void createBorderCu(cu::float4* dev_obstacles,cu::cudaExtent res, int borderPosX, int borderNegX,
															int borderPosY, int borderNegY,
															int borderPosZ, int borderNegZ);

extern "C" void addColliderCu(cu::float4* dev_obstacles, float radius, cu::float3 position,
							cu::cudaExtent res, cu::float3 vel);

extern "C" void advectVelCu(cu::float4* dev_vel, float timestep, float velDamp, cu::float3 invGridSize,
							cu::cudaExtent res);

extern "C" void advectDensCu(float* dev_dens, float timestep, float densDis, cu::float3 invGridSize,
							cu::cudaExtent res);

extern "C" void addDensCu(float* dev_dens, float timestep, float radius, float amount, cu::float3 position,
						  cu::cudaExtent res);

extern "C" void addDensBuoyCu(cu::float4* dev_vel, float timestep, float densBuoyStrength, cu::float3 densBuoyDir,
							  cu::cudaExtent res);

extern "C" void calcNoiseCu(float* dev_noise, cu::cudaExtent res, cu::float3 fluidSize, float yOffset,
									int noiseOct, float noiseLacun, float gain, float noiseFreq, float noiseAmp);

extern "C" void addNoiseCu(cu::float4* dev_vel, float timestep, float noiseStr, cu::cudaExtent res);

extern "C" void vorticityCu(cu::float4* dev_vort, cu::cudaExtent res, cu::float3 invCellSize);

extern "C" void vortConfCu(cu::float4* dev_vel, float timestep, float vortConf, cu::cudaExtent res);

extern "C" void divergenceCu(float* dev_div, cu::cudaExtent res, cu::float3 invCellSize);

extern "C" void jacobiCu(float* dev_pressure, float alpha, float rBeta, cu::cudaExtent res);

extern "C" void projectionCu(cu::float4* dev_vel, cu::cudaExtent res, cu::float3 invCellSize);

extern "C" void addFluidForceKernelCu(cu::float3* vel, cu::float3* pos, cu::cudaExtent gres, cu::float3 invSize, float strength,
									  float dt, int nParts, int trailLength, int leads);




#endif  // __DATABLOCK_H__