#include "fbo.h"
#include <GL/glew.h>


Fbo::Fbo() {

	glGenFramebuffersEXT(1, &fboId);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboId);
}

Fbo::~Fbo(){

	glDeleteFramebuffersEXT(1, &fboId);

}


void Fbo::attachTex(GLuint tex) {

	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, tex, 0);

}

void Fbo::bind() {

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboId);

}

void Fbo::unbind() {

#ifdef HOUDINI
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 10);
#else
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
#endif

}

void Fbo::checkValid(){

	GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
	if(status != GL_FRAMEBUFFER_COMPLETE_EXT)
		int fboUsed = false;

}