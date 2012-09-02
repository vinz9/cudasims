#include <GL/glew.h>

#include <GL/glut.h>
#include <GL/glext.h>

#include "vhParticlesSystem.h"

#include <time.h>


int VHParticlesSystem::numSystems = -1;
VHParticlesSystem* VHParticlesSystem::systemsList[10];

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

inline float frandCenter()
{
    return (frand()*2.0 - 1.0);
}

VHParticlesSystem::VHParticlesSystem(){

	numSystems += 1;
	id = numSystems;
	systemsList[id] = this;

	nParts = -1;
	dt = 1.0/30.0;

	posVbo = 0;
	colourVbo = 0;

	fluidSolver = NULL;

	emitters = new ParticlesEmitter[1];

	nEmit = 1;
	emitters[0].amount = 2000;
	emitters[0].posX = 0;
	emitters[0].posY = 0;
	emitters[0].posZ = 0;
	emitters[0].radVelAmp = 0.0;
	emitters[0].radius = 0.5; // 0.5

	emitters[0].velX = 0;
	emitters[0].velY = 0;
	emitters[0].velZ = 0;

	emitters[0].noiseVelAmpX = 0;
	emitters[0].noiseVelAmpY = 0;
	emitters[0].noiseVelAmpZ = 0;

	partsLife = 4.0;
	partsLifeVar = 0.0;	//1
	velDamp = 0.0;	//0.1
	gravityStrength = 1.0;
	gravityDir = cu::make_float3(0,-1.0,0);
	fluidStrength = 0.0;

	noiseAmp = cu::make_float3(0,0,0); //cu::make_float3(2,2,2);
	noiseOffset = cu::make_float3(0,0,0);
	noiseOct = 1;
	noiseLac = 2;
	noiseFreq = 1;

	pointSize = 1.0;
	opacity = 0.05;
	startColor = cu::make_float3(1,1,1);
	endColor = cu::make_float3(1,1,1);

}

VHParticlesSystem::~VHParticlesSystem(){

	id -= 1;

	freeParticlesSystem();

}

void VHParticlesSystem::initParticlesSystem(int maxParts){

	nParts = maxParts;

	lockOpenGLContext();

	/*AUX_RGBImageRec *pTextureImage = auxDIBImageLoad( "C:\\particle.bmp" );

    if( pTextureImage != NULL )
	{
        glGenTextures( 1, &g_textureID );

		glBindTexture( GL_TEXTURE_2D, g_textureID );

		glTexParameteri( GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR );
		glTexParameteri( GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR );

		glTexImage2D( GL_TEXTURE_2D, 0, 3, pTextureImage->sizeX, pTextureImage->sizeY, 0,
				     GL_RGB, GL_UNSIGNED_BYTE, pTextureImage->data );
	}

	if( pTextureImage )
	{
		if( pTextureImage->data )
			free( pTextureImage->data );

		free( pTextureImage );
	}*/


	glGenBuffersARB(1, &posVbo);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, posVbo);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(cu::float3)*nParts, 0, GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cu::cutilSafeCall(cu::cudaGraphicsGLRegisterBuffer(&posVboRes, posVbo, cu::cudaGraphicsMapFlagsWriteDiscard));


	glGenBuffersARB(1, &colourVbo);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, colourVbo);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(cu::float4)*nParts, 0, GL_DYNAMIC_DRAW_ARB);
	//glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(cu::float4)*2000000, 0, GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cu::cutilSafeCall(cu::cudaGraphicsGLRegisterBuffer(&colourVboRes, colourVbo, cu::cudaGraphicsMapFlagsWriteDiscard));

	unlockOpenGLContext();

	cu::cutilSafeCall(cu::cudaMalloc((void**)&dev_pos,sizeof(cu::float3)*nParts));
	cu::cudaMalloc((void**)&dev_vel,sizeof(cu::float3)*nParts);
	cu::cudaMalloc((void**)&dev_colour,sizeof(cu::float4)*nParts);

	cu::cudaMalloc((void**)&dev_age,sizeof(float)*nParts);
	cu::cudaMalloc((void**)&dev_life,sizeof(float)*nParts);

	cu::cudaMalloc((void**)&tempdev_pos,sizeof(cu::float3)*nParts);
	cu::cudaMalloc((void**)&tempdev_vel,sizeof(cu::float3)*nParts);
	cu::cudaMalloc((void**)&tempdev_age,sizeof(float)*nParts);
	cu::cudaMalloc((void**)&tempdev_life,sizeof(float)*nParts);

	host_pos = new float[maxParts*3];

	index = 0;

	time = 0;

	resetParticles();


}

void VHParticlesSystem::freeParticlesSystem(){

	if(nParts == -1)
		return;

	delete host_pos;

	cu::cudaFree((void**)&dev_pos);
	cu::cudaFree((void**)&dev_vel);
	cu::cudaFree((void**)&dev_colour);
	cu::cudaFree((void**)&dev_life);

	cu::cudaFree((void**)&tempdev_pos);
	cu::cudaFree((void**)&tempdev_vel);
	cu::cudaFree((void**)&tempdev_age);
	cu::cudaFree((void**)&tempdev_life);

	lockOpenGLContext();

	cu::cudaGraphicsUnregisterResource(posVboRes);
	cu::cudaGraphicsUnregisterResource(colourVboRes);

	glDeleteTextures(1, &g_textureID);
	glDeleteBuffersARB(1, &posVbo);
	glDeleteBuffersARB(1, &colourVbo);

	unlockOpenGLContext();

}

void VHParticlesSystem::changeMaxParts(int maxParts){

	freeParticlesSystem();
	initParticlesSystem(maxParts);


}

void VHParticlesSystem::resetParticles(){

	if (nParts == -1)
		return;

	resetParticlesCu(dev_pos, dev_vel, dev_age, dev_life, nParts);
}


void VHParticlesSystem::emitParticles(){

	if (nParts == -1)
		return;

	if (partsLifeVar > partsLife)
		partsLifeVar = partsLife;

	for (int i = 0; i<nEmit; i++) {

		int num = (int)((float)emitters[i].amount*dt);

		cu::float3* pos = new cu::float3[num];
		cu::float3* vel = new cu::float3[num];
		float* age = new float[num];
		float* life = new float[num];

		float x,y,z = 0;

		ParticlesEmitter curEmit = emitters[i];

		for (int j = 0; j<num; j++) {


			do {
				x = frandCenter();
				y = frandCenter();
				z = frandCenter();

			} while (x*x + y*y + z*z > 1.0);

				pos[j].x = curEmit.posX + curEmit.radius * x;
				pos[j].y = curEmit.posY + curEmit.radius * y;
				pos[j].z = curEmit.posZ + curEmit.radius * z;

			//age[j] = -1.0;
			life[j] = partsLife - (partsLifeVar * frand());

			index++;
			if(index>nParts) {
				index = 0;

			}
		}

		/*cu::cudaMemcpy(dev_pos + index, pos, sizeof(cu::float3)*num,cu::cudaMemcpyHostToDevice);
		cu::cudaMemcpy(dev_age + index, age, sizeof(float)*num,cu::cudaMemcpyHostToDevice);
		cu::cudaMemcpy(dev_life + index, life, sizeof(float)*num,cu::cudaMemcpyHostToDevice);*/

		cu::cudaMemcpy(tempdev_pos, pos, sizeof(cu::float3)*num,cu::cudaMemcpyHostToDevice);
		cu::cudaMemcpy(tempdev_life, life, sizeof(float)*num,cu::cudaMemcpyHostToDevice);


		initNewParticlesCu(tempdev_pos, tempdev_vel, tempdev_age, tempdev_life,
							cu::make_float3(curEmit.posX,curEmit.posY,curEmit.posZ),
							cu::make_float3(curEmit.velX,curEmit.velY,curEmit.velZ),
							curEmit.radVelAmp,
							cu::make_float3(curEmit.noiseVelAmpX,curEmit.noiseVelAmpY,curEmit.noiseVelAmpZ),
							cu::make_float3(curEmit.noiseVelOffsetX,curEmit.noiseVelOffsetY,curEmit.noiseVelOffsetZ),
							curEmit.noiseVelOct, curEmit.noiseVelLac, curEmit.noiseVelFreq,
							0, time, num);

		cu::cudaMemcpy(dev_pos + index, tempdev_pos, sizeof(cu::float3)*num,cu::cudaMemcpyDeviceToDevice);
		cu::cudaMemcpy(dev_vel + index, tempdev_vel, sizeof(cu::float3)*num,cu::cudaMemcpyDeviceToDevice);
		cu::cudaMemcpy(dev_age + index, tempdev_age, sizeof(float)*num,cu::cudaMemcpyDeviceToDevice);
		cu::cudaMemcpy(dev_life + index, tempdev_life, sizeof(float)*num,cu::cudaMemcpyDeviceToDevice);

		delete pos;
		delete vel;
		delete age;
		delete life;

	}

}

void VHParticlesSystem::updateParticles(){

	if (nParts == -1)
		return;

	time = time+dt;

	dampVelCu(dev_vel, velDamp, dt, nParts);

	if(fluidStrength != 0 && fluidSolver != NULL) {

		cu::float3 invSize = cu::make_float3(1.0/fluidSolver->fluidSize.x,1.0/fluidSolver->fluidSize.y,
											1.0/fluidSolver->fluidSize.z);

		bindVelTex(fluidSolver->velArray);
		addFluidForceKernelCu(dev_vel, dev_pos, fluidSolver->res, invSize, 1.0, dt,  nParts);
		unbindVelTex();

	}

	if(gravityStrength != 0)
		addGravityCu(dev_vel, gravityDir, gravityStrength, dt, nParts);

	if(noiseAmp.x != 0 || noiseAmp.y != 0 || noiseAmp.z != 0)
		addTurbulenceCu(dev_vel, dev_pos, noiseAmp, noiseOffset, noiseOct, noiseLac, noiseFreq, dt, nParts);

	integrateParticlesCu(dev_pos, dev_vel, dev_age, dev_life, dev_colour, opacity, startColor, endColor, dt, nParts);


}

void VHParticlesSystem::updateVBO(cu::float3* destPos, cu::float4* destColour){

	if (nParts == -1)
		return;

	cu::cudaMemcpy(destPos,dev_pos,sizeof(cu::float3)*nParts,cu::cudaMemcpyDeviceToDevice);
	cu::cudaMemcpy(destColour,dev_colour,sizeof(cu::float4)*nParts,cu::cudaMemcpyDeviceToDevice);

}

void VHParticlesSystem::draw(){

	if (nParts == -1)
		return;

	glEnable(GL_TEXTURE_2D);

	glBindTexture( GL_TEXTURE_2D, g_textureID );

	float quadratic[] =  { 1.0f, 0.0f, 0.01f };
    glPointParameterfvARB( GL_POINT_DISTANCE_ATTENUATION_ARB, quadratic );

    //glPointParameterfARB( GL_POINT_FADE_THRESHOLD_SIZE_ARB, 60.0f );

    //glPointParameterfARB( GL_POINT_SIZE_MIN_ARB, 1.0f );
    //glPointParameterfARB( GL_POINT_SIZE_MAX_ARB, 10.0f );

    glTexEnvf( GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE );

	glPointSize(pointSize);

    glEnable( GL_POINT_SPRITE_ARB );

	cu::float3 *dPosPtr;

	cu::cutilSafeCall(cu::cudaGraphicsMapResources(1, &posVboRes, 0));
	size_t num_bytes_pos;
	cu::cutilSafeCall(cu::cudaGraphicsResourceGetMappedPointer((void **)&dPosPtr, &num_bytes_pos, posVboRes));

	cu::float4 *dColourPtr;

	cu::cutilSafeCall(cu::cudaGraphicsMapResources(1, &colourVboRes, 0));
	size_t num_bytes_col;
	cu::cutilSafeCall(cu::cudaGraphicsResourceGetMappedPointer((void **)&dColourPtr, &num_bytes_col, colourVboRes));


	updateVBO(dPosPtr,dColourPtr);

	cu::cutilSafeCall(cu::cudaGraphicsUnmapResources(1, &posVboRes, 0));
	cu::cutilSafeCall(cu::cudaGraphicsUnmapResources(1, &colourVboRes, 0));

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, posVbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, colourVbo);
	glColorPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);

	glDrawArrays(GL_POINTS, 0, nParts);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);


	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	glDisable( GL_POINT_SPRITE_ARB );

	glDisable(GL_TEXTURE_2D);
	glBindTexture( GL_TEXTURE_2D, 0 );


}

void VHParticlesSystem::lockOpenGLContext(){
}

void VHParticlesSystem::unlockOpenGLContext(){
}
