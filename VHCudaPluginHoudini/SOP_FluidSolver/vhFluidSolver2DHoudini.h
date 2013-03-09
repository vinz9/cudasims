#ifndef __VHFLUIDSOLVER2DHOUD_H__
#define __VHFLUIDSOLVER2DHOUD_H__

#include <RE/RE_Render.h>
#include "../../CudaCommon/CudaFluidSolver2D/vhFluidSolver.h"


struct VHFluidSolver2DHoudini : VHFluidSolver {

	void lockOpenGLContext();
	void unlockOpenGLContext();


};

#endif  // __DATABLOCK_H__