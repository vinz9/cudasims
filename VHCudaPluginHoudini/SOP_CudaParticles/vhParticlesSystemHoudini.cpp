#include "vhParticlesSystemHoudini.h"

void VHParticlesSystemHoudini::lockOpenGLContext(){

	RE_Render* myRender = RE_Render::getRenderContext(2);
	myRender->lockContextForRender();

}

void VHParticlesSystemHoudini::unlockOpenGLContext(){

	RE_Render* myRender = RE_Render::getRenderContext(2);
	myRender->unlockContextAfterRender();

}