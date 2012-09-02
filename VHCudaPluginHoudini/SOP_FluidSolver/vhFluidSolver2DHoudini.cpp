#include "vhFluidSolver2DHoudini.h"

void VHFluidSolver2DHoudini::lockOpenGLContext(){

	RE_Render* myRender = RE_Render::getRenderContext(2);
	myRender->lockContextForRender();

}

void VHFluidSolver2DHoudini::unlockOpenGLContext(){

	RE_Render* myRender = RE_Render::getRenderContext(2);
	myRender->unlockContextAfterRender();

}