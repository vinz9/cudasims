#include <GL/glew.h>

#include <GL/glut.h>
#include <GL/glext.h>

#include <direct.h>

#include "vhParticlesRender.h"
#include "vhParticlesSystem.h"

VHParticlesRender::VHParticlesRender(VHParticlesSystem* currPSys) {

	pSys = currPSys;

	/*displayMode = SHADOWED_SPRITES;
	blendingMode = ALPHA;
	sortParts = 1;*/

	displayMode = POINTS;
	blendingMode = ADD;
	sortParts = 0;

	displayLightBuffer = 0;
	displayVectors = 0;

	pointSize = 1.0;
	lineWidth = 1.0;


	lightPos = vec3f(5,5,-5);
	lightTarget = vec3f(0,0,0);
	lightColor = vec3f(1,1,1);
	colorAttenuation = vec3f(1,1,1);

	shadowAlpha = 0.3;

	nSlices = 128;
	resMul = 1.0;
	width = -1;
	height = -1;
	lightBufferSize = 256;

	doBlur = 1;
	blurRadius = 0.1;

	id1 = 0;
	srcLightTexture = 0;

	spritePath = NULL;

}

VHParticlesRender::~VHParticlesRender(){

	//clearParticlesRender();

}

void VHParticlesRender::initParticlesRender(){


		  // Create the CUDPP radix sort
    CUDPPConfiguration sortConfig;
    sortConfig.algorithm = CUDPP_SORT_RADIX;
    sortConfig.datatype = CUDPP_FLOAT;
    sortConfig.op = CUDPP_ADD;
    sortConfig.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
    cudppPlan(&m_sortHandle, sortConfig, pSys->nParts, 1, 0);

	//-----shaders

	/*char currentPath[_MAX_PATH];
	getcwd(currentPath, _MAX_PATH);
	printf("Path : %s", currentPath);*/

	simpleSpriteProg = new GLSLProgram("sprite.vs", "sprite.gs", "simpleSprite.ps");
	shadowedSpriteProg = new GLSLProgram("sprite.vs", "sprite.gs", "shadowedSprite.ps");
	shadowMapSpriteProg = new GLSLProgram("sprite.vs", "sprite.gs", "ShadowMapSprite.ps");
	displayTexProg = new GLSLProgram("passThru.vs", "texture2D.ps");
	blurProg = new GLSLProgram("passThru.vs", "blur.ps");

	if(spritePath)
		loadSprite(spritePath);

	initFbos(width, height, true);
}

void VHParticlesRender::clearParticlesRender(){

	cudppDestroyPlan(m_sortHandle);

	delete simpleSpriteProg;
	delete shadowedSpriteProg;
	delete shadowMapSpriteProg;
	delete displayTexProg;
	delete blurProg;

	TextureManager::Inst()->UnloadTexture(id1);

	glDeleteTextures(2, lightTex);
    glDeleteTextures(1, &imageTex);

	delete imageFbo;
	delete lightFbo;

}

void VHParticlesRender::initFbos(int newWidth, int newHeight, bool init){

	width = newWidth;
	height = newHeight;

	if(init==false) {
		glDeleteTextures(2, lightTex);
		glDeleteTextures(1, &imageTex);

		delete imageFbo;
		delete lightFbo;
	}

	GLint format = GL_RGBA16F_ARB;
	GLenum status;

    imageTex = createTexture(GL_TEXTURE_2D, width*resMul, height*resMul, format, GL_RGBA);

	//if(init)
	imageFbo = new Fbo();

	imageFbo->attachTex(imageTex);
	imageFbo->checkValid();

    lightTex[0] = createTexture(GL_TEXTURE_2D, lightBufferSize, lightBufferSize, format, GL_RGBA);
	lightTex[1] = createTexture(GL_TEXTURE_2D, lightBufferSize, lightBufferSize, format, GL_RGBA);

	//if(init)
	lightFbo = new Fbo();

	lightFbo->attachTex(lightTex[0]);
	lightFbo->checkValid();

	Fbo::unbind();

}

void VHParticlesRender::loadSprite(char* path){

	TextureManager::Inst()->LoadTexture(path, id1, GL_BGRA, GL_RGBA);

}

GLuint VHParticlesRender::createTexture(GLuint target, int w, int h, GLenum internalformat, GLenum format)
{
    GLuint texid;
    glGenTextures(1, &texid);
    glBindTexture(target, texid);

    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_FLOAT, 0);
    return texid;
}


void VHParticlesRender::calcVectors()
{
    // get model view matrix
    glGetFloatv(GL_MODELVIEW_MATRIX, (float *) modelView.get_value());

    // calculate eye space light vector
    lightVector = normalize(lightPos);
    lightPosEye = modelView * vec4f(lightPos, 1.0);

    // calculate half-angle vector between view and light
    viewVector = -vec3f(modelView.get_row(2));
    if (dot(viewVector, lightVector) > 0) {
        halfVector = normalize(viewVector + lightVector);
        invertedView = false;
    } else {
        halfVector = normalize(-viewVector + lightVector);
        invertedView = true;
    }

    // calculate light view matrix
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    gluLookAt(lightPos[0], lightPos[1], lightPos[2], 
              lightTarget[0], lightTarget[1], lightTarget[2],
              0.0, 1.0, 0.0);

    // calculate light projection matrix
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluPerspective(45.0, 1.0, 1.0, 200.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, (float *) lightView.get_value());
    glGetFloatv(GL_PROJECTION_MATRIX, (float *) lightProj.get_value());

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // construct shadow matrix
    matrix4f scale;
    scale.set_scale(vec3f(0.5, 0.5, 0.5));
    matrix4f translate;
    translate.set_translate(vec3f(0.5, 0.5, 0.5));

    shadowMatrix = translate * scale * lightProj * lightView * inverse(modelView);

    // calc object space eye position
    eyePos = inverse(modelView) * vec4f(0.0, 0.0, 0.0, 1.0);

    // calc half vector in eye space
    halfVectorEye = modelView * vec4f(halfVector, 0.0);

}

// draw points using given shader program
void VHParticlesRender::drawPointSprites(int start, int count)
{
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);  // don't write depth
    glEnable(GL_BLEND);

	glEnable(GL_TEXTURE_2D);

	pSys->posVbo->bind();
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);


	pSys->colourVbo->bind();
	glColorPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);

	pSys->indexVbo->bind();

	glDrawElements(GL_POINTS, count, GL_UNSIGNED_INT, (void*) (start*sizeof(unsigned int)) );

	pSys->indexVbo->unbind();

	pSys->posVbo->unbind();

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	glDisable(GL_TEXTURE_2D);


    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

void VHParticlesRender::drawSlice(int i)
{
	imageFbo->bind();
    glViewport(0, 0, width*resMul, height*resMul);

	//int db;
	//glGetIntegerv ( GL_DRAW_BUFFER, &db );


    if (invertedView) {
        // front-to-back
        glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_ONE);
    } else {
        // back-to-front
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    }

	shadowedSpriteProg->enable();
	shadowedSpriteProg->setUniform1f("pointRadius",pointSize);
    shadowedSpriteProg->bindTexture("sDiffuseMap", TextureManager::Inst()->m_texID[id1],GL_TEXTURE_2D,0);
	shadowedSpriteProg->bindTexture("shadowTex", lightTex[srcLightTexture], GL_TEXTURE_2D, 1);


	drawPointSprites(i*batchSize, batchSize);

	shadowedSpriteProg->disable();

}


// draw slice of particles from light's point of view
void VHParticlesRender::drawSliceLightView(int i)
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadMatrixf((GLfloat *) lightView.get_value());

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadMatrixf((GLfloat *) lightProj.get_value());


	lightFbo->bind();
    glViewport(0, 0, lightBufferSize, lightBufferSize);

	//int db;
	//glGetIntegerv ( GL_DRAW_BUFFER, &db );

    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);

	shadowMapSpriteProg->enable();
	shadowMapSpriteProg->setUniform1f("pointRadius",pointSize);
	shadowMapSpriteProg->setUniform4f("uShadowColor", colorAttenuation.x,colorAttenuation.y,colorAttenuation.z,shadowAlpha);

	shadowMapSpriteProg->bindTexture("sDiffuseMap", TextureManager::Inst()->m_texID[id1],GL_TEXTURE_2D,0);

    drawPointSprites(i*batchSize, batchSize);
	shadowMapSpriteProg->disable();
    

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

void VHParticlesRender::drawSlices()
{
    
	batchSize = pSys->nParts / nSlices;

    glClearColor(0.0, 0.0, 0.0, 0.0);
  
	lightFbo->bind();
	glClearColor(1.0 - lightColor[0], 1.0 - lightColor[1], 1.0 - lightColor[2], 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

	imageFbo->bind();
	glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

	/*int at;
	glGetIntegerv ( GL_ACTIVE_TEXTURE, &at );

	int mm;
	glGetIntegerv ( GL_MATRIX_MODE, &mm );

	GLfloat texMat[16];
	glGetFloatv(GL_TEXTURE_MATRIX, texMat);*/

    glActiveTexture(GL_TEXTURE0);
    glMatrixMode(GL_TEXTURE);
	//glPushMatrix();
    glLoadMatrixf((GLfloat *) shadowMatrix.get_value());

	//---

    for(int i=0; i<nSlices; i++) {
        // draw slice from camera view, sampling light buffer
        drawSlice(i);
        // draw slice from light view to light buffer, accumulating shadows
        drawSliceLightView(i);

		if (doBlur) {
            blurLightBuffer();
        }

    }

    glActiveTexture(GL_TEXTURE0);
    glMatrixMode(GL_TEXTURE);
	//glPopMatrix();
    glLoadIdentity();

	//GLfloat texMat2[16];
	//glGetFloatv(GL_TEXTURE_MATRIX, texMat2);

	glMatrixMode(GL_MODELVIEW);
}

void VHParticlesRender::blurLightBuffer()
{

	lightFbo->bind();
	lightFbo->attachTex(lightTex[1 - srcLightTexture]);


    glViewport(0, 0, lightBufferSize, lightBufferSize);

    blurProg->enable();
    blurProg->bindTexture("tex", lightTex[srcLightTexture], GL_TEXTURE_2D, 0);
    blurProg->setUniform2f("texelSize", 1.0 / (float) lightBufferSize, 1.0 / (float) lightBufferSize);
    blurProg->setUniform1f("blurRadius", blurRadius);
    glDisable(GL_DEPTH_TEST);

	drawQuad();

    blurProg->disable();

    srcLightTexture = 1 - srcLightTexture;

}

void VHParticlesRender::drawVector(vec3f v)
{
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3fv((float *) &v[0]);
    glEnd();
}

// render vectors to screen for debugging
void VHParticlesRender::debugVectors()
{
    glColor3f(1.0, 1.0, 0.0);
    drawVector(lightVector);

	glColor3f(0.0, 1.0, 0.0);
    drawVector(viewVector);

	glColor3f(0.0, 0.0, 1.0);
	drawVector(-viewVector);

	glColor3f(1.0, 0.0, 0.0);
	drawVector(halfVector);

	/*glColor3f(1.0, 1.0, 0.0);
	 glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(lightVector.x,lightVector.y,lightVector.z);
    glEnd();

    glColor3f(0.0, 1.0, 0.0);
    //drawVector(viewVector);
	 glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(viewVector.x,viewVector.y,viewVector.z);
    glEnd();

    glColor3f(0.0, 0.0, 1.0);
	glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(-viewVector.x,-viewVector.y,-viewVector.z);
    glEnd();
    //drawVector(-viewVector);

    glColor3f(1.0, 0.0, 0.0);

	glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(halfVector.x,halfVector.y,halfVector.z);
    glEnd();
    //drawVector(halfVector);


	printf("light : %f, %f, %f \n", lightVector.x, lightVector.y, lightVector.z);
	printf("view : %f, %f, %f \n", viewVector.x, viewVector.y, viewVector.z);
	printf("half : %f, %f, %f \n", halfVector.x, halfVector.y, halfVector.z);
	printf("-------------------------\n");

	float lx,ly,lz;
	float vx,vy,vz;
	float hx, hy, hz;

	lx = lightVector.x;
	ly = lightVector.y;
	lz = lightVector.z;

	vx = viewVector.x;
	vy = viewVector.y;
	vz = viewVector.z;

	hx = halfVector.x;
	hy = halfVector.y;
	hz = halfVector.z;



	

	glColor3f(1.0, 1.0, 0.0);

	 glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(lx,ly,lz);
    glEnd();

    glColor3f(0.0, 1.0, 0.0);
	 glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(vx,vy,vz);
    glEnd();

    glColor3f(1.0, 0.0, 0.0);

	glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(hx,hy,hz);
    glEnd();*/


}

void VHParticlesRender::drawQuad() {

	glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
    glEnd();

}

void VHParticlesRender::draw(){

	if (pSys->nParts == -1)
		return;

	if(displayMode == SHADOWED_SPRITES) {

			calcVectors();
			cu::float3 halfVec = cu::make_float3(halfVector.x,halfVector.y,halfVector.z); 

			calcDepthCu(pSys->dev_pos, pSys->dev_keys, pSys->dev_indices, halfVec, pSys->nParts);

			if (sortParts)
				cudppSort(m_sortHandle, pSys->dev_keys, pSys->dev_indices, 32, pSys->nParts);
	}

	if((displayMode == SPRITES || displayMode == POINTS) && sortParts) {

		glGetFloatv(GL_MODELVIEW_MATRIX, (float *) modelView.get_value());
		 viewVector = -vec3f(modelView.get_row(2));

		cu::float3 viewVec = cu::make_float3(viewVector.x, viewVector.y, viewVector.y);
		//printf("view vec : %f, %f, %f \n", viewVector.x, viewVector.y, viewVector.z);
		calcDepthCu(pSys->dev_pos, pSys->dev_keys, pSys->dev_indices, viewVec, pSys->nParts);

		cudppSort(m_sortHandle, pSys->dev_keys, pSys->dev_indices, 32, pSys->nParts);


	}

	pSys->posVbo->map();
	pSys->colourVbo->map();
	pSys->indexVbo->map();

	pSys->updateVBOs();

	pSys->posVbo->unmap();
	pSys->colourVbo->unmap();
	pSys->indexVbo->unmap();

	switch (displayMode) {

		case POINTS:

			glPointSize(pointSize);

			glDisable(GL_DEPTH_TEST);

			glEnable(GL_BLEND);

			pSys->posVbo->bind();
			glVertexPointer(3, GL_FLOAT, 0, 0);
			glEnableClientState(GL_VERTEX_ARRAY);

			pSys->colourVbo->bind();
			glColorPointer(4, GL_FLOAT, 0, 0);
			glEnableClientState(GL_COLOR_ARRAY);

			if (blendingMode == ADD) {
				glBlendFunc( GL_SRC_ALPHA, GL_ONE );
				glDrawArrays(GL_POINTS, 0, pSys->nParts);
			} else {
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				if(sortParts){
					pSys->indexVbo->bind();
					glDrawElements(GL_POINTS, pSys->nParts, GL_UNSIGNED_INT, 0);
					pSys->indexVbo->unbind();
				} else {
					glDrawArrays(GL_POINTS, 0, pSys->nParts);
				}
			}


			pSys->posVbo->unbind();

			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_COLOR_ARRAY);

			glDisable(GL_BLEND);

			break;

		case LINES:

			glDisable(GL_DEPTH_TEST);

			glEnable(GL_BLEND);
			glBlendFunc( GL_SRC_ALPHA, GL_ONE );

			pSys->posVbo->bind();
			glVertexPointer(3, GL_FLOAT, 0, 0);
			glEnableClientState(GL_VERTEX_ARRAY);

			pSys->colourVbo->bind();
			glColorPointer(4, GL_FLOAT, 0, 0);
			glEnableClientState(GL_COLOR_ARRAY);

			glLineWidth(lineWidth);
			for (int i = 0; i<pSys->nLeadParts; i++) {
				glDrawArrays(GL_LINE_STRIP, i*pSys->trailLength, pSys->trailLength);
			}

			pSys->posVbo->unbind();

			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_COLOR_ARRAY);

			glDisable(GL_BLEND);

			break;

		case SPRITES:


			glDisable(GL_DEPTH_TEST);

			glEnable(GL_TEXTURE_2D);

			glEnable(GL_BLEND);
			glBlendFunc( GL_SRC_ALPHA, GL_ONE );

			pSys->posVbo->bind();
			glVertexPointer(3, GL_FLOAT, 0, 0);
			glEnableClientState(GL_VERTEX_ARRAY);

			pSys->colourVbo->bind();
			glColorPointer(4, GL_FLOAT, 0, 0);
			glEnableClientState(GL_COLOR_ARRAY);

			simpleSpriteProg->enable();
			simpleSpriteProg->setUniform1f("pointRadius",pointSize);
			simpleSpriteProg->bindTexture("sDiffuseMap",TextureManager::Inst()->m_texID[id1],GL_TEXTURE_2D,0);

			if (blendingMode == ADD) {
				glBlendFunc( GL_SRC_ALPHA, GL_ONE );
				glDrawArrays(GL_POINTS, 0, pSys->nParts);
			} else {
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

				if(sortParts){
					pSys->indexVbo->bind();
					glDrawElements(GL_POINTS, pSys->nParts, GL_UNSIGNED_INT, 0);
					pSys->indexVbo->unbind();
				} else {
					glDrawArrays(GL_POINTS, 0, pSys->nParts);
				}
			}

			simpleSpriteProg->disable();

			pSys->posVbo->unbind();

			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_COLOR_ARRAY);

			glDisable(GL_BLEND);

			glDisable(GL_TEXTURE_2D);

			break;

		case SHADOWED_SPRITES :

			GLfloat currentViewport[4];
			glGetFloatv(GL_VIEWPORT, currentViewport);

			if(width != currentViewport[2] || height != currentViewport[3])
				initFbos(currentViewport[2],currentViewport[3], false);

			drawSlices();

			//glutReportErrors();

			Fbo::unbind();


			glViewport(0, 0, width, height);
			glDisable(GL_DEPTH_TEST);
			glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
			glEnable(GL_BLEND);

			int mm;
			glGetIntegerv ( GL_MATRIX_MODE, &mm );

			displayTexProg->enable();
			displayTexProg->bindTexture("tex", imageTex, GL_TEXTURE_2D, 0);
		    
			drawQuad();

			displayTexProg->disable();

			if(displayLightBuffer) {

				displayTexProg->bindTexture("tex", lightTex[srcLightTexture], GL_TEXTURE_2D, 0);
				glViewport(0, 0, lightBufferSize, lightBufferSize);
				drawQuad();
				displayTexProg->disable();
			}

			//calcVectors();

			glViewport(0, 0, width, height);

 			if (displayVectors) {
				debugVectors();
			}

			glutReportErrors();

			break;

	}

}