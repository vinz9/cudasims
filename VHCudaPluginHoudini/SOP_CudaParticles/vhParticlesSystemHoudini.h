#ifndef __VHPARTICLESSYTEMHOUD_H__
#define __VHPARTICLESSYTEMHOUD_H__

#include <RE/RE_Render.h>
#include "../../CudaCommon/CudaParticlesSystem/vhParticlesSystem.h"


struct VHParticlesSystemHoudini : VHParticlesSystem {

	void lockOpenGLContext();
	void unlockOpenGLContext();

};

#endif  // __DATABLOCK_H__