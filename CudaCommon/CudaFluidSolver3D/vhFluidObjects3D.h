#ifndef __VHFLUIDOBJECTS_H__
#define __VHFLUIDOBJECTS_H__

struct VHFluidEmitter {

	float posX;
	float posY;
	float posZ;

	float radius;
	float amount;
	
};

struct VHFluidCollider {

	float posX;
	float posY;
	float posZ;

	float oldPosX;
	float oldPosY;
	float oldPosZ;

	float radius;
	
};

#endif 