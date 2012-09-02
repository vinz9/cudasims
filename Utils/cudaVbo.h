#ifndef CUDAVBO_H
#define CUDAVBO_H

#include <stdio.h>

namespace cu{
	#include <cuda_runtime_api.h>
	//#include <cuda_gl_interop.h>
	#include <vector_functions.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>
}

typedef int GLsizei;

template <class T> 
class CudaVbo
{
public:

	CudaVbo(int num, int elmt);
	~CudaVbo();

	T* dPtr;

	int element;

	void map();
	void unmap();

	void bind();
	void unbind();
  
	unsigned int vboId;
	struct cu::cudaGraphicsResource *vboRes;

};

#include "cudaVbo.cpp"

#endif
