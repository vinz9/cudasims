#ifndef __VHPARTICLESRENDER_H__
#define __VHPARTICLESRENDER_H__

#include <stdlib.h>
#include <stdio.h> //very important
#include <cmath>

#include "../../Utils/GLSLProgram.h"
#include "../../Utils/TextureManager.h"
#include "../../Utils/cudaVbo.h"
#include "../../Utils/fbo.h"
//#include "../../Utils/textfile.h"

#include "nvMath.h"

using namespace nv;

#include "vhParticlesObjects.h"
#include "cudpp/cudpp.h"

typedef unsigned int GLuint;

struct VHParticlesSystem;

struct VHParticlesRender {

	VHParticlesRender(VHParticlesSystem* pSys);
	~VHParticlesRender();

	VHParticlesSystem* pSys;

	CUDPPHandle m_sortHandle;

	int displayMode;
	int blendingMode;
	int sortParts;
	int displayLightBuffer;
	int displayVectors;

	float pointSize;
	float lineWidth;

	char* spritePath;

	vec3f				lightPos, lightTarget, lightColor, colorAttenuation;
	float shadowAlpha;

	int doBlur;
	float blurRadius;

	int nSlices;
	float resMul;
	int width, height;
	int lightBufferSize;


	//----

	enum displayModeEnum {
        POINTS,
		LINES,
		SPRITES,
        SHADOWED_SPRITES,
        NUM_MODES
    };

	enum blendingEnum {
		ADD,
		ALPHA,
		NUM_BLENDMODES

	};

	GLuint imageTex;
	GLuint lightTex[2];
	int      srcLightTexture;

	Fbo*	imageFbo;
	Fbo*	lightFbo;

	GLuint id1;
	GLSLProgram         *simpleSpriteProg;
	GLSLProgram         *shadowedSpriteProg;
	GLSLProgram         *shadowMapSpriteProg;
	GLSLProgram         *displayTexProg;
	GLSLProgram         *blurProg;

	int batchSize;

	matrix4f            modelView;
    vec3f               viewVector;
    vec4f               eyePos;
	bool                invertedView;

	matrix4f			lightView, lightProj, shadowMatrix;

	vec3f               lightVector, halfVector;
	vec4f               halfVectorEye, lightPosEye;

	void initParticlesRender();
	void clearParticlesRender();
	void initFbos(int newWidth, int newHeight, bool init);

	void loadSprite(char* path);
	GLuint createTexture(GLuint target, int w, int h, GLenum internalformat, GLenum format);

	void calcVectors();

	void drawSlices();
	void drawSlice(int i);
	void drawSliceLightView(int i);
	void drawPointSprites(int start, int count);

	void blurLightBuffer();

	void debugVectors();

	void drawVector(vec3f v);
	void drawQuad();

	void draw();

	//virtual void lockOpenGLContext();
	//virtual void unlockOpenGLContext();

};


#endif  // __DATABLOCK_H__