#include <GL/glew.h>

#include <GL/glut.h>
#include <GL/glext.h>

#include "vhParticlesSystem.h"

#include <time.h>

#include "../../Utils/GLSLProgram.h"
#include "../../Utils/textfile.h"


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

	emitters = new ParticlesEmitter[1];

	nEmit = 1;
	emitters[0].amount = 2000;
	emitters[0].pos = cu::make_float3(0,0,0);
	emitters[0].radVelAmp = 0.0;
	emitters[0].radius = 0.5; // 0.5

	emitters[0].vel = cu::make_float3(0,0,0);

	emitters[0].noiseVelAmp = cu::make_float3(0,0,0);

	partsLife = 4.0;
	partsLifeVar = 0.0;

	inheritVel = 0.0;
	inheritAge = 0.0;


	leadsForces[0] = new DampingForce(this);
	leadsForces[1] = new GravityForce(this);
	leadsForces[2] = new TurbulenceForce(this);
	leadsForces[3] = new FluidForce(this);

	nLeadsForces = 4;

	trailsForces[0] = new DampingForce(this);
	trailsForces[1] = new GravityForce(this);
	trailsForces[2] = new TurbulenceForce(this);
	trailsForces[3] = new FluidForce(this);

	nTrailsForces = 4;

	preview = 0;

	opacity = 0.05;
	startColor = cu::make_float3(1,1,1);
	endColor = cu::make_float3(1,1,1);

	pRend = new VHParticlesRender(this);


}

VHParticlesSystem::~VHParticlesSystem(){

	id -= 1;

	freeParticlesSystem();
	delete pRend;

}

void VHParticlesSystem::initParticlesSystem(int newLeadParts, int newTrailLength){

	nLeadParts = newLeadParts;
	trailLength = newTrailLength;
	nParts = nLeadParts*trailLength;

	lockOpenGLContext();

	//----- vbos

	posVbo = new CudaVbo<cu::float3>(nParts, 0);
	colourVbo = new CudaVbo<cu::float4>(nParts, 0);
	indexVbo = new CudaVbo<unsigned int>(nParts, 1);

	pRend->initParticlesRender();

	unlockOpenGLContext();

	cu::cutilSafeCall(cu::cudaMalloc((void**)&dev_pos,sizeof(cu::float3)*nParts));
	cu::cudaMalloc((void**)&dev_vel,sizeof(cu::float3)*nParts);
	cu::cudaMalloc((void**)&dev_colour,sizeof(cu::float4)*nParts);

	cu::cudaMalloc((void**)&dev_age,sizeof(float)*nParts);
	cu::cudaMalloc((void**)&dev_life,sizeof(float)*nParts);
	cu::cudaMalloc((void**)&dev_opafix,sizeof(char)*nParts);

	cu::cudaMalloc((void**)&tempdev_pos,sizeof(cu::float3)*nParts);
	cu::cudaMalloc((void**)&tempdev_vel,sizeof(cu::float3)*nParts);
	cu::cudaMalloc((void**)&tempdev_age,sizeof(float)*nParts);
	cu::cudaMalloc((void**)&tempdev_life,sizeof(float)*nParts);

	cu::cudaMalloc((void**)&dev_indices,sizeof(unsigned int)*nParts);
	cu::cudaMalloc((void**)&dev_keys,sizeof(float)*nParts);

	host_pos = new float[nParts*3];

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
	cu::cudaFree((void**)&dev_opafix);


	cu::cudaFree((void**)&tempdev_pos);
	cu::cudaFree((void**)&tempdev_vel);
	cu::cudaFree((void**)&tempdev_age);
	cu::cudaFree((void**)&tempdev_life);

	cu::cudaFree((void**)&dev_indices);
	cu::cudaFree((void**)&dev_keys);

	lockOpenGLContext();

	pRend->clearParticlesRender();

	delete posVbo;
	delete colourVbo;
	delete indexVbo;


	unlockOpenGLContext();

}

void VHParticlesSystem::changeMaxParts(int newLeadParts, int newTrailLength){

	freeParticlesSystem();
	initParticlesSystem(newLeadParts,newTrailLength);


}

void VHParticlesSystem::resetParticles(){

	if (nParts == -1)
		return;

	resetParticlesCu(dev_pos, dev_vel, dev_age, dev_life, dev_opafix, nParts);
}


void VHParticlesSystem::emitParticles(){

	if (nParts == -1)
		return;

	if (partsLifeVar > partsLife)
		partsLifeVar = partsLife;


	for (int i = 0; i<nEmit; i++) {

		float rate = ((float)emitters[i].amount)*dt;
		int numLeadParts = 0;

		if (rate < 1) {
			if ((int)(time/dt) % ((int)(1/rate)) != 0)
				return;
			else
				numLeadParts = 1;
		} else {
			numLeadParts = (int)((float)emitters[i].amount)*dt;
		}

		int num = numLeadParts*trailLength;

		cu::float3* pos = new cu::float3[num];
		cu::float3* vel = new cu::float3[num];
		float* age = new float[num];
		float* life = new float[num];

		float x,y,z = 0;

		ParticlesEmitter curEmit = emitters[i];

		for (int j = 0; j<numLeadParts; j++) {


			do {
				x = frandCenter();
				y = frandCenter();
				z = frandCenter();

			} while (x*x + y*y + z*z > 1.0);

				pos[j*trailLength].x = curEmit.pos.x + curEmit.radius * x;
				pos[j*trailLength].y = curEmit.pos.y + curEmit.radius * y;
				pos[j*trailLength].z = curEmit.pos.z + curEmit.radius * z;

			life[j*trailLength] = partsLife - (partsLifeVar * frand());

			index = index + trailLength;
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
							curEmit.pos, curEmit.vel,curEmit.radVelAmp,
							curEmit.noiseVelAmp, curEmit.noiseVelOffset,
							curEmit.noiseVelOct, curEmit.noiseVelLac, curEmit.noiseVelFreq,
							0, time, numLeadParts, trailLength);

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

	for(int i = 0; i<nLeadsForces; i++)
		leadsForces[i]->applyForce(1);


	if (trailLength > 1) {

		int drawLines;

		if(pRend->displayMode == VHParticlesRender::LINES)
			drawLines = 1;
		else
			drawLines = 0;

		emitTrailsCu(dev_pos, dev_vel, dev_age, dev_life, dev_opafix, inheritVel, inheritAge, partsLife, dt, nLeadParts, trailLength, drawLines);

		for(int i = 0; i<nTrailsForces; i++)
			trailsForces[i]->applyForce(0);


		}

	integrateParticlesCu(dev_pos, dev_vel, dev_age, dev_life, partsLife, dev_colour, opacity,
						startColor, endColor, dev_opafix, dt, nParts, trailLength);

	/*lockOpenGLContext();

	GLfloat modelView[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);*/

    // calculate half-angle vector between view and light
	//cu::float3 viewVector = cu::make_float3(modelView[8],modelView[9],modelView[10]) ;


	//calcDepthCu(dev_pos, dev_keys, dev_indices, viewVector, nParts);

	/*if(pRend->displayMode == VHParticlesRender::SHADOWED_SPRITES) {

			pRend->calcVectors();
			cu::float3 halfVec = cu::make_float3(pRend->halfVector.x,pRend->halfVector.y,pRend->halfVector.z); 

			calcDepthCu(dev_pos, dev_keys, dev_indices, halfVec, nParts);

			if (pRend->sortParts)
				cudppSort(m_sortHandle, dev_keys, dev_indices, 32, nParts);
	}

	if((pRend->displayMode == VHParticlesRender::SPRITES || pRend->displayMode == VHParticlesRender::POINTS) && pRend->sortParts) {

		//cu::float3 viewVec = cu::make_float3(modelView[8],modelView[9],modelView[10]) ;
		cu::float3 viewVec = cu::make_float3(pRend->viewVector.x, pRend->viewVector.y, pRend->viewVector.y);
		calcDepthCu(dev_pos, dev_keys, dev_indices, viewVec, nParts);

		cudppSort(m_sortHandle, dev_keys, dev_indices, 32, nParts);


	}*/

	//unlockOpenGLContext();



}

void VHParticlesSystem::updateVBOs(){

	if (nParts == -1)
		return;

	cu::cudaMemcpy(posVbo->dPtr,dev_pos,sizeof(cu::float3)*nParts,cu::cudaMemcpyDeviceToDevice);
	cu::cudaMemcpy(colourVbo->dPtr,dev_colour,sizeof(cu::float4)*nParts,cu::cudaMemcpyDeviceToDevice);
	cu::cudaMemcpy(indexVbo->dPtr,dev_indices,sizeof(unsigned int)*nParts,cu::cudaMemcpyDeviceToDevice);

}


void VHParticlesSystem::draw(){

	if (nParts == -1)
		return;


	pRend->draw();
}

void VHParticlesSystem::lockOpenGLContext(){
}

void VHParticlesSystem::unlockOpenGLContext(){
}
