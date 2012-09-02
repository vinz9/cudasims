#ifndef __VHPARTICLESOBJECTS_H__
#define __VHPARTICLESOBJECTS_H__

struct ParticlesEmitter {

	float posX;
	float posY;
	float posZ;

	float radius;
	float amount;

	float velX;
	float velY;
	float velZ;

	float radVelAmp;

	float noiseVelAmpX;
	float noiseVelAmpY;
	float noiseVelAmpZ;

	float noiseVelOffsetX;
	float noiseVelOffsetY;
	float noiseVelOffsetZ;

	int noiseVelOct;
	float noiseVelLac;
	float noiseVelFreq;

	
};

struct ParticlesCollider {

	float posX;
	float posY;
	float posZ;

	float oldPosX;
	float oldPosY;
	float oldPosZ;

	float radius;
	
};

#endif 