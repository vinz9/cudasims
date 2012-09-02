#ifndef __FLUID2D_H__
#define __FLUID2D_H__

#include "vhObjects.h"
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

struct VHFluidSolver {

	VHFluidSolver();
	~VHFluidSolver();

	static VHFluidSolver* solverList[10];
	static int numSolvers;

	int displayX, displayY;

	uint gl_Tex;
	uint gl_SliceTex;

	uint pbo;     // OpenGL pixel buffer object
	struct cu::cudaGraphicsResource *cuda_pbo_resource;

	void initPixelBuffer();


	int id;

	int preview;
	int previewType;
	float bounds;

	int f;
	int nEmit;
	FluidEmitter* emitters;

	int nColliders;
	Collider* colliders;

	int fps;
	int substeps;
	int jacIter;

	cu::int2 res;

	cu::float2 fluidSize;

	float densDis;
	float densBuoyStrength;
	cu::float2 densBuoyDir;

	float velDamp;
	float vortConf;

	float noiseStr;
	float noiseFreq;
	int noiseOct;
	float noiseLacun;
	float noiseSpeed;
	float noiseAmp;

	int colOutput;

	int borderPosX;
	int borderNegX;
	int borderPosY;
	int borderNegY;

	float			*host_dens;
	float			*host_vel;

	cu::float4		*output_display;

	float			*dev_noise;
	cu::float2          *dev_vel;
	float           *dev_dens;
	float           *dev_pressure;
	float           *dev_div;
	float           *dev_vort;
	cu::float4			*dev_obstacles;

	cu::cudaArray *noiseArray;
	cu::cudaArray *velArray;
	cu::cudaArray *densArray;
	cu::cudaArray *pressureArray;
	cu::cudaArray *divArray;
	cu::cudaArray *vortArray;
	cu::cudaArray *obstArray;


    //cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;

	void changeFluidRes(int x, int y);
	void initFluidSolver(int x, int y);
	void clearFluid();
	void solveFluid();
	void resetFluid();
	void drawFluid(float fluidRotX, float fluidRotY, float fluidRotZ,
					float fluidPosX, float fluidPosY, float fluidPosZ);

	long domainSize( void ) const { return res.x * res.y * sizeof(float); }

	virtual void lockOpenGLContext();
	virtual void unlockOpenGLContext();

};

extern "C" void setup2DTexturesCu();
extern "C" void bind2DTexturesCu(cu::cudaArray* noiseArray, cu::cudaArray* velArray, cu::cudaArray* densArray,
								cu::cudaArray* pressureArray, cu::cudaArray* divArray, cu::cudaArray* vortArray,
								cu::cudaArray* obstArray);
extern "C" void unbind2DTexturesCu();

extern "C" void createBorder2DCu(cu::float4* dev_obstacles, cu::int2 res, int borderPosX, int borderNegX,
																int borderPosY, int borderNegY);

extern "C" void addCollider2DCu(cu::float4* dev_obstacles, float radius, cu::float2 position, cu::int2 res, cu::float2 vel);
extern "C" void advectVel2DCu(cu::float2* dev_vel, float timestep, float velDamp, cu::float2 invGridSize, cu::int2 res);
extern "C" void advectDens2DCu(float* dev_dens, float timestep, float densDis, cu::float2 invGridSize, cu::int2 res);
extern "C" void addDens2DCu(float* dev_dens, float timestep, float radius, float amount, cu::float2 position, cu::int2 res);
extern "C" void addDensBuoy2DCu(cu::float2* dev_vel, float timestep, float densBuoyStrength, cu::float2 densBuoyDir, cu::int2 res);
extern "C" void calcNoise2DCu(float* dev_noise, cu::int2 res, cu::float2 fluidSize, float offset,
							int noiseOct, float noiseLacun, float noiseFreq, float noiseAmp);
extern "C" void addNoise2DCu(cu::float2* dev_vel, float timestep, float noiseStr, cu::int2 res);
extern "C" void vorticity2DCu(float* dev_vort, cu::int2 res, cu::float2 invCellSize);
extern "C" void vortConf2DCu(cu::float2* dev_vel, float timestep, float vortConfStrength, cu::int2 res);
extern "C" void divergence2DCu(float* dev_div, cu::int2 res, cu::float2 invCellSize);
extern "C" void jacobi2DCu(float* dev_pressure, float alpha, float rBeta, cu::int2 res);
extern "C" void projection2DCu(cu::float2* dev_vel, cu::int2 res, cu::float2 invCellSize);

extern "C" void render2DFluidCu(cu::float4* d_output, int previewType, float maxBounds, cu::int2 res,
								float* dev_dens, cu::float2* dev_vel, float* dev_noise, float* dev_pressure,
								float* dev_vort, cu::float4* dev_obstacles);

extern "C" void free_d_permCu();

#endif  // __DATABLOCK_H__
