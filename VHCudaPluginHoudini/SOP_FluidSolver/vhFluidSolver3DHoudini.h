#ifndef __VHFLUIDSOLVER3DHOUD_H__
#define __VHFLUIDSOLVER3DHOUD_H__

#include <RE/RE_Render.h>
#include <UT/UT_XformOrder.h>
#include "../../CudaCommon/CudaFluidSolver3D/vhFluidSolver3D.h"


struct VHFluidSolver3DHoudini : VHFluidSolver3D {

	void lockOpenGLContext();
	void unlockOpenGLContext();

	void calculateTransRot(float* modelviewH, cu::float3* trans, cu::float3* rot);

};

#endif  // __DATABLOCK_H__

