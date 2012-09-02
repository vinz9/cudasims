#include <GL/glew.h>

#include <GL/glut.h>
#include <GL/glext.h>

#include "vhFluidSolver.h"

int VHFluidSolver::numSolvers = -1;
VHFluidSolver* VHFluidSolver::solverList[10];

VHFluidSolver::VHFluidSolver() {

	numSolvers += 1;
	solverList[numSolvers] = this;
	id = numSolvers;

	f = 0;
	fps = 30;
	substeps = 1;
	jacIter = 50;

	fluidSize = cu::make_float2(10.0,10.0);
	res = cu::make_int2(-1,-1);

	borderPosX = 1;
	borderNegX = 1;
	borderPosY = 1;
	borderNegY = 1;

	densDis = 0.0;
	densBuoyStrength = 1; //1
	densBuoyDir = cu::make_float2(0.0,1.0);

	velDamp = 0.01;
	vortConf = 5.0; //5

	noiseStr = 1.0; //1.0
	noiseFreq = 1.0;
	noiseOct = 3;
	noiseLacun = 4.0;
	noiseSpeed = 0.01;
	noiseAmp = 0.5;

	emitters = new FluidEmitter[1];
	colliders = new Collider[1];

	nEmit = 1;
	emitters[0].amount = 1;
	emitters[0].posX = 0;
	emitters[0].posY = -4; //-4
	emitters[0].radius = 0.5;

	nColliders = 0;
	colliders[0].posX = 0;
	colliders[0].posY = 0;
	colliders[0].radius = 1;

	preview = 1;
	previewType = 0;
	bounds = 1;

	displayX = displayY = 256;
	pbo = 0;

}

VHFluidSolver::~VHFluidSolver() {

	numSolvers -= 1;

	clearFluid();

	delete emitters;
	delete colliders;

	lockOpenGLContext();

	cu::cudaGraphicsUnregisterResource(cuda_pbo_resource);

	glDeleteBuffersARB(1, &pbo);
	glDeleteTextures(1, &gl_Tex);

	unlockOpenGLContext();

}

void VHFluidSolver::changeFluidRes(int x, int y){

	clearFluid();
	initFluidSolver(x, y);

}

void VHFluidSolver::initFluidSolver(int x, int y){

	res.x = x;
	res.y = y;

	initPixelBuffer();

	host_dens = new float[res.x*res.y* sizeof(float)];

	cu::cudaChannelFormatDesc descFloat = cu::cudaCreateChannelDesc<float>();
	cu::cudaChannelFormatDesc descFloat2 = cu::cudaCreateChannelDesc<cu::float2>();
	cu::cudaChannelFormatDesc descFloat4 = cu::cudaCreateChannelDesc<cu::float4>();


	cu::cudaMalloc( (void**)&dev_noise,sizeof(float)*res.x*res.y);
	cu::cudaMallocArray(&noiseArray, &descFloat, res.x, res.y);


	cu::cudaMalloc( (void**)&dev_vel,sizeof(cu::float2)*res.x*res.y);
	cu::cudaMallocArray(&velArray, &descFloat2, res.x, res.y);

	cu::cudaMalloc( (void**)&dev_dens,sizeof(float)*res.x*res.y);
	cu::cudaMallocArray(&densArray, &descFloat, res.x, res.y);

	cu::cudaMalloc( (void**)&dev_pressure,sizeof(float)*res.x*res.y);
	cu::cudaMallocArray(&pressureArray, &descFloat, res.x, res.y);

	cu::cudaMalloc( (void**)&dev_div,sizeof(float)*res.x*res.y);
	cu::cudaMallocArray(&divArray, &descFloat, res.x, res.y);

	cu::cudaMalloc( (void**)&dev_vort,sizeof(float)*res.x*res.y);
	cu::cudaMallocArray(&vortArray, &descFloat, res.x, res.y);

	cu::cudaMalloc( (void**)&dev_obstacles,sizeof(cu::float4)*res.x*res.y);
	cu::cudaMallocArray(&obstArray, &descFloat4, res.x, res.y);

	setup2DTexturesCu();

}

void VHFluidSolver::clearFluid(){

	if (res.x != -1) {	

		delete host_dens;
		
		cu::cudaFree(dev_noise);
		cu::cudaFreeArray(noiseArray);

		cu::cudaFree(dev_vel);
		cu::cudaFreeArray(velArray);

		cu::cudaFree(dev_dens);
		cu::cudaFreeArray(densArray);

		cu::cudaFree(dev_pressure);
		cu::cudaFreeArray(pressureArray);

		cu::cudaFree(dev_div);
		cu::cudaFreeArray(divArray);

		cu::cudaFree(dev_vort);
		cu::cudaFreeArray(vortArray);

		cu::cudaFree(dev_obstacles);
		cu::cudaFreeArray(obstArray);

		free_d_permCu();
		

		/*if (colOutput==1)
			cu::cudaFree(output_display);*/

	}

}

void VHFluidSolver::solveFluid(){

	bind2DTexturesCu(noiseArray, velArray, densArray, pressureArray, divArray, vortArray, obstArray);

	float timestep = 1.0/(fps*substeps);
	float radius = 0;
	cu::float2 position = cu::make_float2(0,0);
	cu::float2 invGridSize = cu::make_float2(1/fluidSize.x,1/fluidSize.y);
	//float2 invCellSize = make_float2(d->res.x/d->fluidSize.x, d->res.y/d->fluidSize.y);
	cu::float2 invCellSize = cu::make_float2(1.0,1.0);

	float alpha = -(1.0/invCellSize.x*1.0/invCellSize.y);
	float rBeta = 0.25;

	for (int i=0; i<substeps; i++) {

		createBorder2DCu(dev_obstacles, res, borderPosX, borderNegX, borderPosY, borderNegY);

		for (int j=0; j<nColliders; j++) {

			position = cu::make_float2(res.x*0.5+res.x/fluidSize.x*colliders[j].posX,
										res.y*0.5+res.y/fluidSize.y*colliders[j].posY);

			radius = colliders[j].radius*res.x/fluidSize.x;
			cu::float2 vel = cu::make_float2(1.0f/timestep * (colliders[j].posX - colliders[j].oldPosX),
									1.0f/timestep*(colliders[j].posY - colliders[j].oldPosY));

			addCollider2DCu(dev_obstacles, radius, position, res, vel);
		}
	
		
		cu::cudaMemcpyToArray( velArray, 0, 0, dev_vel, res.x*res.y*sizeof(cu::float2), cu::cudaMemcpyDeviceToDevice);
		cu::cudaMemcpyToArray( obstArray, 0, 0, dev_obstacles, res.x*res.y*sizeof(cu::float4), cu::cudaMemcpyDeviceToDevice);
		advectVel2DCu(dev_vel,timestep,velDamp,invGridSize,res);


		cu::cudaMemcpyToArray(velArray, 0, 0, dev_vel, res.x*res.y*sizeof(cu::float2), cu::cudaMemcpyDeviceToDevice);
		cu::cudaMemcpyToArray(densArray, 0, 0, dev_dens, res.x*res.y*sizeof(float), cu::cudaMemcpyDeviceToDevice);
		advectDens2DCu(dev_dens,timestep,densDis,invGridSize,res);
		
		for (int j=0; j<nEmit; j++) {
			position = cu::make_float2(res.x*0.5+res.x/fluidSize.x*emitters[j].posX,
								res.y*0.5+res.y/fluidSize.y*emitters[j].posY);

			radius = emitters[j].radius*res.x/fluidSize.x;
			addDens2DCu(dev_dens,timestep,radius,emitters[j].amount,position,res);
		}

		addDensBuoy2DCu(dev_vel,timestep,densBuoyStrength,cu::make_float2(densBuoyDir.x,densBuoyDir.y),res);

		if(noiseStr != 0) {
			calcNoise2DCu(dev_noise, res, fluidSize, f*noiseSpeed, noiseOct, noiseLacun, noiseFreq, noiseAmp);
			cu::cudaMemcpyToArray(noiseArray, 0, 0, dev_noise, res.x*res.y*sizeof(float), cu::cudaMemcpyDeviceToDevice);
			addNoise2DCu(dev_vel, timestep, noiseStr, res);
		} else {
			cu::cudaMemset(dev_noise,0, sizeof(float) * res.x * res.y);
		}

		if(vortConf != 0) {
			cu::cudaMemcpyToArray(velArray, 0, 0, dev_vel, res.x*res.y*sizeof(cu::float2), cu::cudaMemcpyDeviceToDevice);
			vorticity2DCu(dev_vort, res, invCellSize);

			cu::cudaMemcpyToArray(vortArray, 0, 0, dev_vort, res.x*res.y*sizeof(float), cu::cudaMemcpyDeviceToDevice);
			vortConf2DCu(dev_vel, timestep, vortConf, res);
		}

		cu::cudaMemcpyToArray(velArray, 0, 0, dev_vel, res.x*res.y*sizeof(cu::float2), cu::cudaMemcpyDeviceToDevice);
		divergence2DCu(dev_div,res,invCellSize);

		cu::cudaMemset(dev_pressure,0, sizeof(float) *res.x * res.y);

		cu::cudaMemcpyToArray(divArray, 0, 0, dev_div, res.x*res.y*sizeof(float), cu::cudaMemcpyDeviceToDevice);
		for (int i=0; i<jacIter; i++) {
			cu::cudaMemcpyToArray(pressureArray, 0, 0, dev_pressure, res.x*res.y*sizeof(float), cu::cudaMemcpyDeviceToDevice);
			jacobi2DCu(dev_pressure, alpha, rBeta, res);
		}


		cu::cudaMemcpyToArray(velArray, 0, 0, dev_vel, res.x*res.y*sizeof(cu::float2), cu::cudaMemcpyDeviceToDevice);
		cu::cudaMemcpyToArray(pressureArray, 0, 0, dev_pressure, res.x*res.y*sizeof(float), cu::cudaMemcpyDeviceToDevice);
		projection2DCu(dev_vel,res,invCellSize);
		

	}

	f++;
	
	unbind2DTexturesCu();

}


void VHFluidSolver::resetFluid(){

	cu::cudaMemset(dev_vel,0, sizeof(cu::float2) * res.x * res.y);
	cu::cudaMemset(dev_dens,0, sizeof(float) * res.x * res.y);
	cu::cudaMemset(dev_pressure,0, sizeof(float) * res.x * res.y);
	cu::cudaMemset(dev_div,0, sizeof(float) * res.x * res.y);
	cu::cudaMemset(dev_vort,0, sizeof(float) * res.x * res.y);
	cu::cudaMemset(dev_noise,0, sizeof(float) * res.x * res.y);
	cu::cudaMemset(dev_obstacles,0, sizeof(cu::float4) * res.x * res.y);

	f = 0;

}

void VHFluidSolver::initPixelBuffer(){

	lockOpenGLContext();


	if (pbo) {
		// unregister this buffer object from CUDA C
		cu::cutilSafeCall(cu::cudaGraphicsUnregisterResource(cuda_pbo_resource));

		// delete old buffer
		glDeleteBuffersARB(1, &pbo);
		glDeleteTextures(1, &gl_Tex);
	}

	// create pixel buffer object for display
	glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, displayX*displayY*sizeof(cu::float4), 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// register this buffer object with CUDA
	cu::cutilSafeCall(cu::cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cu::cudaGraphicsMapFlagsWriteDiscard));	

	// create texture for display
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, displayX, displayY, 0, GL_RGBA, GL_FLOAT,  NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	

	unlockOpenGLContext();

}

void VHFluidSolver::drawFluid(float fluidRotX, float fluidRotY, float fluidRotZ,
					float fluidPosX, float fluidPosY, float fluidPosZ){

		float sizeX = fluidSize.x*0.5;
		float sizeY = fluidSize.y*0.5;

		int newResX = res.x;
		int newResY = res.y;

		if(displayX != newResX || displayY != newResY) {
			displayX = newResX;
			displayY = newResY;
			initPixelBuffer();
		}
					
		cu::cutilSafeCall(cu::cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		cu::float4 *d_output;
		size_t num_bytes; 
		cu::cutilSafeCall(cu::cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  cuda_pbo_resource));
		cu::cudaMemset(d_output, 0, displayX*displayY*sizeof(cu::float4));

		render2DFluidCu(d_output, previewType, bounds, res, dev_dens, dev_vel, dev_noise, dev_pressure,
								dev_vort, dev_obstacles);

		cu::cutilSafeCall(cu::cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

		//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glBindTexture(GL_TEXTURE_2D, gl_Tex);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, displayX, displayY, GL_RGBA, GL_FLOAT, 0);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		glEnable(GL_TEXTURE_2D);

		glPushMatrix();
		glTranslatef(fluidPosX,fluidPosY,fluidPosZ);
		glRotatef(fluidRotZ,0,0,1);
		glRotatef(fluidRotY,0,1,0);
		glRotatef(fluidRotX,1,0,0);


		glColor3f(1.0,1.0,1.0);
		glDisable(GL_BLEND);
		glDisable(GL_LIGHTING);

		glBegin( GL_QUADS );
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-sizeX,-sizeY,0.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-sizeX,sizeY,0.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex3f(sizeX,sizeY,0.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex3f(sizeX,-sizeY,0.0f);
		glEnd();

		glDisable(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, 0);

		glBegin( GL_LINE_LOOP );
		glVertex3f(-sizeX,-sizeY,0.0f);
		glVertex3f(-sizeX,sizeY,0.0f);
		glVertex3f(sizeX,sizeY,0.0f);
		glVertex3f(sizeX,-sizeY,0.0f);
		glEnd();

		glPopMatrix();
}

void VHFluidSolver::lockOpenGLContext(){
}

void VHFluidSolver::unlockOpenGLContext(){
}