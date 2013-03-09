#include "initGLCuda.h"

MTypeId     initGLCuda::id( 0x800009 );

void* initGLCuda::creator()
{
	return new initGLCuda();
}

MStatus initGLCuda::initialize()
{

	return MS::kSuccess;
} 

initGLCuda::initGLCuda() {

	cu::cutilSafeCall(cu::cudaThreadExit());

	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
	  std::cout << glewGetErrorString(err) << std::endl;
	}

	cu::cutilSafeCall(cu::cudaGLSetGLDevice( cu::cutGetMaxGflopsDeviceId() ));



}


initGLCuda::~initGLCuda() {

	cu::cutilSafeCall(cu::cudaThreadExit());

}

MStatus initGLCuda::compute (const MPlug& plug, MDataBlock& data) {

	MStatus returnStatus;

	return MS::kSuccess;

}