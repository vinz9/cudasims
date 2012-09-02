#include <GL/glew.h>

#include <GL/glut.h>
#include <GL/glext.h>

#include "vhFluidSolver3D.h"


int VHFluidSolver3D::numSolvers = -1;
VHFluidSolver3D* VHFluidSolver3D::solverList[10];

VHFluidSolver3D::VHFluidSolver3D() {

	numSolvers += 1;
	solverList[numSolvers] = this;
	id = numSolvers;

	preview = 0;
	drawCube = 0;
	opaScale = 1;
	stepMul = 1;
	displayRes = 1;

	doShadows = 0;
	shadowDens = 1;
	shadowStepMul = 2;
	shadowThres = 1;

	displaySlice = 0;
	sliceType = 0;
	sliceAxis = 2;
	slicePos = 0.5;
	sliceBounds = 1;

	lightPos = cu::make_float3(10,10,0);

	f = 0;
	fps = 30;
	substeps = 1;
	jacIter = 30;

	fluidSize = cu::make_float3(10.0,10.0,10.0);

	res.width = -1;
	res.height = -1;
	res.depth = -1;
	
	borderPosX = 1;
	borderNegX = 1;
	borderPosY = 1;
	borderNegY = 1;
	borderPosZ = 1;
	borderNegZ = 1;

	densDis = 0.0;
	densBuoyStrength = 1; //1
	densBuoyDir = cu::make_float3(0.0,1.0,0.0);

	velDamp = 0.01;
	vortConf = 5.0; //5

	noiseStr = 0.0; //1.0
	noiseFreq = 1.0;
	noiseOct = 3;
	noiseLacun = 4.0;
	noiseSpeed = 0.01;
	noiseAmp = 0.5;

	emitters = new VHFluidEmitter[1];
	colliders = new VHFluidCollider[1];

	nEmit = 1;
	emitters[0].amount = 1;
	emitters[0].posX = 0;
	emitters[0].posY = -4; //-4
	emitters[0].posZ = 0;
	emitters[0].radius = 0.5;

	nColliders = 0;
	colliders[0].posX = 0;
	colliders[0].posY = 0;
	colliders[0].posZ = 0;
	colliders[0].radius = 1;

	displayX = displayY = 256;
	displaySliceX = displaySliceY = 60;
	displayEnum = 1;
	//displayEnum = -1;
	pbo = 0;

}

VHFluidSolver3D::~VHFluidSolver3D() {
	numSolvers -= 1;

	clearFluid();

	delete emitters;
	delete colliders;

	lockOpenGLContext();

	cu::cudaGraphicsUnregisterResource(cuda_pbo_resource);

	glDeleteBuffersARB(1, &pbo);
	glDeleteTextures(1, &gl_Tex);
	glDeleteTextures(1, &gl_SliceTex);

	unlockOpenGLContext();

}


void VHFluidSolver3D::changeFluidRes(int x, int y, int z){


	clearFluid();
	initFluidSolver(x, y, z);


}

void VHFluidSolver3D::initFluidSolver(int x, int y, int z){

	initPixelBuffer(true);

	host_dens = new float[x*y*z* sizeof(float)];
	host_vel = new float[x*y*z* sizeof(float)*4];

	res = cu::make_cudaExtent(x, y, z);

	cu::cutilSafeCall(cu::cudaMalloc((void**)&dev_dens, sizeof(float)*res.width * res.height * res.depth));
	cu::cudaMalloc((void**)&dev_noise, sizeof(float)*res.width * res.height * res.depth);
	cu::cudaMalloc((void**)&dev_vel, sizeof(cu::float4)*res.width * res.height * res.depth);
	cu::cudaMalloc((void**)&dev_div, sizeof(float)*res.width * res.height * res.depth);
	cu::cudaMalloc((void**)&dev_pressure, sizeof(float)*res.width * res.height * res.depth);
	cu::cudaMalloc((void**)&dev_obstacles, sizeof(cu::float4)*res.width * res.height * res.depth);
	cu::cudaMalloc((void**)&dev_vort, sizeof(cu::float4)*res.width * res.height * res.depth);

	cu::cudaChannelFormatDesc descFloat4_3d = cu::cudaCreateChannelDesc<cu::float4>();
	cu::cudaChannelFormatDesc descFloat_3d = cu::cudaCreateChannelDesc<float>();
	
	cu::cutilSafeCall(cu::cudaMalloc3DArray(&densArray, &descFloat_3d, res));
	cu::cudaMalloc3DArray(&noiseArray, &descFloat_3d, res);
	cu::cudaMalloc3DArray(&velArray, &descFloat4_3d, res);
	cu::cudaMalloc3DArray(&divArray, &descFloat_3d, res);
	cu::cudaMalloc3DArray(&pressureArray, &descFloat_3d, res);
	cu::cudaMalloc3DArray(&obstaclesArray, &descFloat4_3d, res);
	cu::cudaMalloc3DArray(&vortArray, &descFloat4_3d, res);

	setupTextures();
}

void VHFluidSolver3D::clearFluid() {

	if (res.width != -1) {

		delete host_dens;
		delete host_vel;
		
		cu::cudaFree( dev_noise );
		cu::cudaFree( dev_dens );
		cu::cudaFree( dev_vel );
		cu::cudaFree( dev_div );
		cu::cudaFree( dev_pressure);
		cu::cudaFree( dev_obstacles);
		cu::cudaFree( dev_vort);

		cu::cudaFreeArray(densArray);
		cu::cudaFreeArray(noiseArray);
		cu::cudaFreeArray(velArray);
		cu::cudaFreeArray(divArray);
		cu::cudaFreeArray(pressureArray);
		cu::cudaFreeArray(obstaclesArray);
		cu::cudaFreeArray(vortArray);

		if (colOutput==1) {
			cu::cudaFree( output_display);
			cu::cudaFree( output_display_slice );
		}
	}

}

void VHFluidSolver3D::fieldCopy(void* field, cu::cudaArray* fieldArray, size_t size) {

	cu::cudaMemcpy3DParms copyParams = {0};

	copyParams.srcPtr.ptr   = field;
	copyParams.srcPtr.pitch = res.width*size;
	copyParams.srcPtr.xsize = res.width;
    copyParams.srcPtr.ysize = res.height;

	copyParams.dstArray = fieldArray;
    copyParams.extent   = res;
	copyParams.kind     = cu::cudaMemcpyDeviceToDevice;

	cu::cutilSafeCall(cu::cudaMemcpy3D(&copyParams));

}

void VHFluidSolver3D::solveFluid(){

	bindDensTex(densArray);
	bindNoiseTex(noiseArray);
	bindVelTex(velArray);
	bindDivTex(divArray);
	bindPressureTex(pressureArray);
	bindObstaclesTex(obstaclesArray);
	bindVortTex(vortArray);

	float timestep = 1.0/(fps*substeps);

	cu::float3 invGridSize = cu::make_float3(1/fluidSize.x,1/fluidSize.y,1/fluidSize.z);
	//float2 invCellSize = make_float2(d->res.x/d->fluidSize.x, d->res.y/d->fluidSize.y);
	cu::float3 invCellSize = cu::make_float3(1.0,1.0,1.0);

	float alpha = -(1.0/invCellSize.x*1.0/invCellSize.y*1.0/invCellSize.z);
	float rBeta =  1/6.0;


	for (int i=0; i<substeps; i++) {

		createBorderCu(dev_obstacles,res, borderPosX, borderNegX, borderPosY, borderNegY,
													borderPosZ, borderNegZ);

		float radius = 0;
		cu::float3 position = cu::make_float3(0,0,0);
		cu::float3 vel = cu::make_float3(0,0,0);

		for (int j=0; j<nColliders; j++) {
			position = cu::make_float3(res.width/2+res.width/fluidSize.x*colliders[j].posX,
								res.height/2+res.height/fluidSize.y*colliders[j].posY,
								res.depth/2+res.depth/fluidSize.z*colliders[j].posZ);
			radius = colliders[j].radius*res.width/(fluidSize.x);
			vel = cu::make_float3(1.0f/timestep*(colliders[j].posX - colliders[j].oldPosX),
									1.0f/timestep*(colliders[j].posY - colliders[j].oldPosY),
									1.0f/timestep*(colliders[j].posZ - colliders[j].oldPosZ));
			addColliderCu(dev_obstacles, radius, position, res, vel);
		}

		fieldCopy(dev_obstacles, obstaclesArray, sizeof(cu::float4));
		fieldCopy(dev_vel, velArray, sizeof(cu::float4));
		advectVelCu(dev_vel,timestep,velDamp,invGridSize,res);

		//addVel<<<blocks,threads>>>(d->dev_vel, 1, d->res);

		fieldCopy(dev_vel, velArray, sizeof(cu::float4));
		fieldCopy(dev_dens, densArray, sizeof(float));
		advectDensCu(dev_dens,timestep,densDis,invGridSize,res);

		for (int j=0; j<nEmit; j++) {
			position = cu::make_float3(res.width/2+res.width/fluidSize.x*emitters[j].posX,
									res.height/2+res.height/fluidSize.y*emitters[j].posY,
									res.depth/2+res.depth/fluidSize.z*emitters[j].posZ);

			radius = emitters[j].radius*res.width/(fluidSize.x);
			addDensCu(dev_dens,timestep,radius,emitters[j].amount,position,res);
		}


		fieldCopy(dev_dens, densArray, sizeof(float));
		addDensBuoyCu(dev_vel,timestep,densBuoyStrength,densBuoyDir,res);

		if(noiseStr != 0) {
			calcNoiseCu(dev_noise, res, fluidSize, f*noiseSpeed, noiseOct, noiseLacun, 0.75f, noiseFreq, noiseAmp);
			fieldCopy(dev_noise, noiseArray, sizeof(float));
			addNoiseCu(dev_vel, timestep, noiseStr, res);
		} else {
			cu::cudaMemset(dev_noise,0, sizeof(float) * res.width * res.height * res.depth);
		}

		if(vortConf != 0) {

			fieldCopy(dev_vel, velArray, sizeof(cu::float4));
			vorticityCu(dev_vort, res, invCellSize);

			fieldCopy(dev_vort, vortArray, sizeof(cu::float4));
			vortConfCu(dev_vel, timestep, vortConf, res);
		} else {
			cu::cudaMemset(dev_vort,0, sizeof(cu::float4) * res.width * res.height * res.depth);
		}

		fieldCopy(dev_vel, velArray, sizeof(cu::float4));
		divergenceCu(dev_div,res,invCellSize);

		cu::cudaMemset(dev_pressure,0, sizeof(float) * res.width * res.height * res.depth);

		fieldCopy(dev_div, divArray, sizeof(float));
		for (int i=0; i<jacIter; i++) {
			fieldCopy(dev_pressure, pressureArray, sizeof(float));
			jacobiCu(dev_pressure, alpha, rBeta,res);
		}


		fieldCopy(dev_vel, velArray, sizeof(cu::float4));
		fieldCopy(dev_pressure, pressureArray, sizeof(float));
		projectionCu(dev_vel,res,invCellSize);

		

	}

	unbindDensTex();
	unbindNoiseTex();
	unbindVelTex();
	unbindDivTex();
	unbindPressureTex();
	unbindObstaclesTex();
	unbindVortTex();

	f++;

}


void VHFluidSolver3D::resetFluid(){

	f = 0;

	cu::cudaMemset(dev_dens,0, sizeof(float) * res.width * res.height * res.depth);
	cu::cudaMemset(dev_noise,0, sizeof(float) * res.width * res.height * res.depth);
	cu::cudaMemset(dev_vel,0, sizeof(cu::float4) * res.width * res.height * res.depth);
	cu::cudaMemset(dev_div,0, sizeof(float) * res.width * res.height * res.depth);
	cu::cudaMemset(dev_pressure,0, sizeof(float) * res.width * res.height * res.depth);
	cu::cudaMemset(dev_obstacles,0, sizeof(cu::float4) * res.width * res.height * res.depth);
	cu::cudaMemset(dev_vort,0, sizeof(cu::float4) * res.width * res.height * res.depth);


}

void VHFluidSolver3D::initPixelBuffer(bool initpbo){

	lockOpenGLContext();

	if (initpbo) {

		if (pbo) {
			// unregister this buffer object from CUDA C
			cu::cutilSafeCall(cu::cudaGraphicsUnregisterResource(cuda_pbo_resource));

			// delete old buffer
			glDeleteBuffersARB(1, &pbo);
		}

		// create pixel buffer object for display
		glGenBuffersARB(1, &pbo);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, 512*512*sizeof(cu::float4), 0, GL_STREAM_DRAW_ARB);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		// register this buffer object with CUDA
		cu::cutilSafeCall(cu::cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cu::cudaGraphicsMapFlagsWriteDiscard));	
	}

	if (pbo) {
		glDeleteTextures(1, &gl_Tex);
	}

	// create texture for display
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, displayX, displayY, 0, GL_RGBA, GL_FLOAT,  NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	//}
	

	if (pbo) {
		glDeleteTextures(1, &gl_SliceTex);

	}

	// create texture for display
	glGenTextures(1, &gl_SliceTex);
	glBindTexture(GL_TEXTURE_2D, gl_SliceTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, displaySliceX, displaySliceY, 0, GL_RGBA, GL_FLOAT,  NULL);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, displayX, displayY, 0, GL_RGBA, GL_FLOAT,  NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	unlockOpenGLContext();

}

void VHFluidSolver3D::renderFluid(cu::float4* d_output, double focalLength){

	fieldCopy(dev_dens, densArray, sizeof(float));

	bindDensTex(densArray);

	renderFluidCu(d_output, displayX, displayY, opaScale, res, focalLength, fluidSize, lightPos,
							stepMul, shadowStepMul, shadowThres, shadowDens, doShadows);
	
	unbindDensTex();

}

void VHFluidSolver3D::drawFluid(float fluidRotX, float fluidRotY, float fluidRotZ,
					float fluidPosX, float fluidPosY, float fluidPosZ){

	int newDisplayRes = displayRes;

	if (newDisplayRes != displayEnum) {
		displayEnum = newDisplayRes;

		switch(displayEnum) {

			case 0 :
				displayX = displayY = 128;
				break;
			case 1 :
				displayX = displayY = 256;
				break;
			case 2 :
				displayX = displayY = 512;
				break;
			/*case 3 :
				displayX = displayY = 768;
				break;
			case 4 :
				displayX = displayY = 1024;
				break;*/
		}

		initPixelBuffer(false);
	}

	float sizeX = fluidSize.x*0.5;
	float sizeY = fluidSize.y*0.5;
	float sizeZ = fluidSize.z*0.5;

	GLfloat modelViewH[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, modelViewH);

	cu::float3 trans = cu::make_float3(0,0,0);
	cu::float3 rot = cu::make_float3(0,0,0);
	calculateTransRot(modelViewH, &trans, &rot);
	
	GLfloat modelView[16];
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
		glLoadIdentity();

		glRotatef(-fluidRotX,1,0,0);
		glRotatef(-fluidRotY,0,1,0);
		glRotatef(-fluidRotZ,0,0,1);
		glTranslatef(-fluidPosX,-fluidPosY,-fluidPosZ);

		glTranslatef(-trans.x, -trans.y, -trans.z);
		glRotatef(-rot.x, 1.0, 0.0, 0.0);
		glRotatef(-rot.y, 0.0, 1.0, 0.0);
		glRotatef(-rot.z, 0.0, 0.0, 1.0);

	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	glPopMatrix();

	float invViewMatrix[12];

	invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
	invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
	invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

	copyInvViewMatrix(invViewMatrix, sizeof(cu::float4)*3);

	cu::cutilSafeCall(cu::cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	cu::float4 *d_output;
	size_t num_bytes; 
	cu::cutilSafeCall(cu::cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  cuda_pbo_resource));
	cu::cudaMemset(d_output, 0, displayX*displayY*sizeof(cu::float4));

	GLfloat projH[16];
	glGetFloatv(GL_PROJECTION_MATRIX, projH);

	double focalLength = -projH[5];

	GLfloat viewportH[4];
	glGetFloatv(GL_VIEWPORT, viewportH);

	float ratio = viewportH[3]/viewportH[2];

	renderFluid(d_output, focalLength);

	cu::cutilSafeCall(cu::cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, displayX, displayY, GL_RGBA, GL_FLOAT, 0);
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, displaySliceX, displaySliceY, GL_RGBA, GL_FLOAT, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -1.0, 1.0, 0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	//glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_TEXTURE_2D);

	glColor4f(1.0, 1.0, 1.0,1.0f);

	//glDepthMask( GL_FALSE );

	glDisable(GL_DEPTH_TEST);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f*ratio, -1.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f*ratio, -1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f*ratio, 1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f*ratio, 1.0f);
	glEnd();

	glEnable(GL_DEPTH_TEST);

	//glDepthMask( GL_TRUE );


	glDisable(GL_TEXTURE_2D);
	//glDisable(GL_BLEND);

	glBindTexture(GL_TEXTURE_2D, 0);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

}

void VHFluidSolver3D::drawFluidSlice(float fluidRotX, float fluidRotY, float fluidRotZ,
					float fluidPosX, float fluidPosY, float fluidPosZ){

	float slicePos3D;

	int newResX = res.width;
	int newResY = res.height;
	int newResZ = res.depth;

	if (sliceAxis == 2) {
		slicePos3D = (slicePos-0.5)*fluidSize.z;

		if (displaySliceX != newResX || displaySliceY != newResY) {
			displaySliceX = newResX;
			displaySliceY = newResY;
			initPixelBuffer(false);
		}

	} else if (sliceAxis == 0) {
		slicePos3D = (slicePos-0.5)*fluidSize.x;

		if (displaySliceX != newResZ || displaySliceY != newResY) {
			displaySliceX = newResZ;
			displaySliceY = newResY;
			initPixelBuffer(false);
		}

	} else {
		slicePos3D = (slicePos-0.5)*fluidSize.y;

		if (displaySliceX != newResX || displaySliceY != newResZ) {
			displaySliceX = newResX;
			displaySliceY = newResZ;
			initPixelBuffer(false);
		}
	}

	cu::cutilSafeCall(cu::cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	cu::float4 *d_output;
	size_t num_bytes; 
	cu::cutilSafeCall(cu::cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  cuda_pbo_resource));
	cu::cudaMemset(d_output, 0, displaySliceX*displaySliceY*sizeof(cu::float4));

	//if (fluidInitialized)
		
	renderFluidSliceCu(d_output, res, slicePos, sliceAxis, sliceType, sliceBounds, dev_dens, dev_vel, dev_noise, 
						dev_pressure, dev_vort, dev_obstacles);

	cu::cutilSafeCall(cu::cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, gl_SliceTex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, displaySliceX, displaySliceY, GL_RGBA, GL_FLOAT, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glEnable(GL_TEXTURE_2D);

	glColor3f(1.0, 1.0, 1.0);

	glPushMatrix();
	glTranslatef(fluidPosX,fluidPosY,fluidPosZ);
	glRotatef(fluidRotZ,0,0,1);
	glRotatef(fluidRotY,0,1,0);
	glRotatef(fluidRotX,1,0,0);

	float sizeX = fluidSize.x * 0.5;
	float sizeY = fluidSize.y * 0.5;
	float sizeZ = fluidSize.z * 0.5;

	if (sliceAxis == 2) {

		glBegin( GL_QUADS );
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-sizeX,-sizeY,slicePos3D);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-sizeX,sizeY,slicePos3D);
		glTexCoord2f(1.0f, 1.0f); glVertex3f(sizeX,sizeY,slicePos3D);
		glTexCoord2f(1.0f, 0.0f); glVertex3f(sizeX,-sizeY,slicePos3D);
		glEnd();

	} else if (sliceAxis == 0) {

		glBegin( GL_QUADS );
		glTexCoord2f(0.0f, 0.0f); glVertex3f(slicePos3D,-sizeY,-sizeZ);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(slicePos3D,sizeY,-sizeZ);
		glTexCoord2f(1.0f, 1.0f); glVertex3f(slicePos3D,sizeY,sizeZ);
		glTexCoord2f(1.0f, 0.0f); glVertex3f(slicePos3D,-sizeY,sizeZ);
		glEnd();
	}

	else {

		glBegin( GL_QUADS );
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-sizeX,slicePos3D,-sizeZ);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-sizeX,slicePos3D,sizeZ);
		glTexCoord2f(1.0f, 1.0f); glVertex3f(sizeX,slicePos3D,sizeZ);
		glTexCoord2f(1.0f, 0.0f); glVertex3f(sizeX,slicePos3D,-sizeZ);
		glEnd();
	}

	glPopMatrix();


	glDisable(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);

}

void VHFluidSolver3D::lockOpenGLContext(){
}

void VHFluidSolver3D::unlockOpenGLContext(){
}

void VHFluidSolver3D::calculateTransRot(float* modelviewH, cu::float3 *trans, cu::float3 *rot){

}