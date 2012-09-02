#ifndef CUDAVBO_CPP
#define CUDAVBO_CPP

//#include <GL/glew.h>
#include "cudaVbo.h"

template <class T> 
CudaVbo<T>::CudaVbo(int num, int elmt) {

	element = elmt;

	glGenBuffersARB(1, &vboId);

	if(!element) {
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboId);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(T) * num, 0, GL_DYNAMIC_DRAW_ARB);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	} else {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, vboId);
		glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, sizeof(T) * num, 0, GL_DYNAMIC_DRAW_ARB);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	cu::cutilSafeCall(cu::cudaGraphicsGLRegisterBuffer(&vboRes, vboId, cu::cudaGraphicsMapFlagsWriteDiscard));
}

template <class T> 
CudaVbo<T>::~CudaVbo(){

	cu::cudaGraphicsUnregisterResource(vboRes);
	glDeleteBuffersARB(1, &vboId);

}

template <class T> 
void CudaVbo<T>::map(){

	cu::cutilSafeCall(cu::cudaGraphicsMapResources(1, &vboRes, 0));
	size_t num_bytes_pos; 
	cu::cutilSafeCall(cu::cudaGraphicsResourceGetMappedPointer((void **)&dPtr, &num_bytes_pos, vboRes));

}

template <class T>
void CudaVbo<T>::unmap(){

	cu::cutilSafeCall(cu::cudaGraphicsUnmapResources(1, &vboRes, 0));

}

template <class T>
void CudaVbo<T>::bind(){

	if(!element)
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboId);
	else
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, vboId);

}

template <class T>
void CudaVbo<T>::unbind(){

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

}

#endif