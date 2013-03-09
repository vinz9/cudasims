#ifndef FBO_H
#define FBO_H

#include <stdio.h>


typedef unsigned int GLuint;

class Fbo
{
public:

	Fbo();
	~Fbo();

	GLuint fboId;

	void attachTex(GLuint tex);

	void bind();
	static void unbind();

	void checkValid();
  

};

#endif
