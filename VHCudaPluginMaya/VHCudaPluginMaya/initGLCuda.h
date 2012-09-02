#ifndef INITGLCUDA_H
#define INITGLCUDA_H

#include <GL/glew.h>

#include <maya/MPxNode.h>

namespace cu{
	#include <cuda_runtime_api.h>
	#include <cuda_gl_interop.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>
}

class initGLCuda : public MPxNode
{
public:
						initGLCuda();
	virtual				~initGLCuda(); 

	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );



	static	void*		creator();
	static	MStatus		initialize();
	//void postConstructor();
 
	static	MTypeId		id;


};

#endif