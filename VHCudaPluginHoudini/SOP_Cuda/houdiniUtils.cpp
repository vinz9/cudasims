#include "houdiniUtils.h"

void lockOpenGLContext(){

	RE_Render* myRender = RE_Render::getRenderContext(2);
	myRender->lockContextForRender();

}

void unlockOpenGLContext(){

	RE_Render* myRender = RE_Render::getRenderContext(2);
	myRender->unlockContextAfterRender();

}