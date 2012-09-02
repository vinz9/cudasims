#ifndef __VHOBJECTS_H__
#define __VHOBJECTS_H__


struct FluidEmitter {

	float posX;
	float posY;

	float radius;
	float amount;
	
};

struct Collider {

	float posX;
	float posY;

	float oldPosX;
	float oldPosY;

	float radius;
	
};

#endif 