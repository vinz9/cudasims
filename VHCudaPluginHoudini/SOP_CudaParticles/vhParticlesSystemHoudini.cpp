#include "vhParticlesSystemHoudini.h"

void VHParticlesSystemHoudini::lockOpenGLContext(){

	//warning not called during VHParticlesSystem constructor

	RE_Render* myRender = RE_Render::getRenderContext(2);
	myRender->lockContextForRender();

}

void VHParticlesSystemHoudini::unlockOpenGLContext(){

	RE_Render* myRender = RE_Render::getRenderContext(2);
	myRender->unlockContextAfterRender();

}