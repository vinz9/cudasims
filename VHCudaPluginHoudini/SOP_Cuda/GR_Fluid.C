#pragma warning(disable : 4018)
#pragma warning(disable : 4244)
#pragma warning(disable : 4996)
#pragma warning(disable : 4800)
#pragma warning(disable : 4312)

#include <GL/glew.h>

#include <GL/glut.h>
#include <GL/glext.h>

#include "GR_Fluid.h"
#include "SOP_FluidSolver2D.h"
#include "SOP_FluidSolver3D.h"

void GR_Fluid::initPixelBuffer(bool initpbo){

	if (initpbo) {

		if (pbo) {
			// unregister this buffer object from CUDA C
			cu::cutilSafeCall(cu::cudaGraphicsUnregisterResource(cuda_pbo_resource));

			// delete old buffer
			glDeleteBuffersARB(1, &pbo);
			//glDeleteTextures(1, &gl_Tex);
			//glDeleteTextures(1, &gl_SliceTex);
		}

		// create pixel buffer object for display
		glGenBuffersARB(1, &pbo);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, 512*512*sizeof(cu::float4), 0, GL_STREAM_DRAW_ARB);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		// register this buffer object with CUDA
		cu::cutilSafeCall(cu::cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cu::cudaGraphicsMapFlagsWriteDiscard));	
	}

	if (pbo) {
		glDeleteTextures(1, &gl_Tex);
	}

	// create texture for display
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, displayX, displayY, 0, GL_RGBA, GL_FLOAT,  NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	//}
	

	if (pbo) {
		glDeleteTextures(1, &gl_SliceTex);

	}

	// create texture for display
	glGenTextures(1, &gl_SliceTex);
	glBindTexture(GL_TEXTURE_2D, gl_SliceTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, displaySliceX, displaySliceY, 0, GL_RGBA, GL_FLOAT,  NULL);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, displayX, displayY, 0, GL_RGBA, GL_FLOAT,  NULL);
	glBindTexture(GL_TEXTURE_2D, 0);



}

GR_Fluid::GR_Fluid() {

	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
	  fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
	}

	cu::cutilSafeCall(cu::cudaGLSetGLDevice(cu::cutGetMaxGflopsDeviceId()));
	//displayX = displayY = -1;
	displayX = displayY = 256;
	displaySliceX = displaySliceY = 60;
	displayEnum = 1;
	//displayEnum = -1;
	pbo = 0;
	initPixelBuffer(true);

}

GR_Fluid::~GR_Fluid() {

	cu::cudaGraphicsUnregisterResource(cuda_pbo_resource);

	cu::cudaThreadExit();

	glDeleteBuffersARB(1, &pbo);
	glDeleteTextures(1, &gl_Tex);

}

int GR_Fluid::preview(const GU_Detail *gdp) {

	GEO_AttributeHandle fluidAh= gdp->getDetailAttribute("cudaFluidPreview");
	fluidAh.setElement(gdp);

	GEO_AttributeHandle fluid3DAh= gdp->getDetailAttribute("cudaFluid3DPreview");
	fluid3DAh.setElement(gdp);

	if (fluidAh.getI() || fluid3DAh.getI())
		return 1;
	else
		return 0;

}


void GR_Fluid::renderWire(GU_Detail *gdp, RE_Render &ren, const GR_AttribOffset & /*ptinfo*/,
		    const GR_DisplayOption *dopt, float /*lod*/, const GU_PrimGroupClosure * /*hidden_geometry*/)
{

	GEO_AttributeHandle fluidAh= gdp->getDetailAttribute("cudaFluidPreview");
	fluidAh.setElement(gdp);

	GEO_AttributeHandle fluid3DAh= gdp->getDetailAttribute("cudaFluid3DPreview");
	fluid3DAh.setElement(gdp);

	GEO_AttributeHandle fluid3DSliceAh= gdp->getDetailAttribute("sliceDisplay");
	fluid3DSliceAh.setElement(gdp);

	GEO_AttributeHandle fluidAh= gdp->getDetailAttribute("cudaFluidPreview");
	fluidAh.setElement(gdp);

	GEO_AttributeHandle fluidIdAh= gdp->getDetailAttribute("solverId");
	fluidIdAh.setElement(gdp);

	if (fluidAh.getI()== 1) {

		VHFluidSolver* currSolver = VHFluidSolver::solverList[fluidIdAh.getI()];

		UT_Vector4 fluidPos(0,0,0);
		UT_Vector3D fluidRot(0,0,0);

		if(gdp->volumeCount() == 1) {
			GEO_Primitive* pprim = gdp->primitives().head();
			GU_PrimVolume* volume = (GU_PrimVolume *)pprim;
			UT_Matrix4 fluidRotMat;
			volume->getTransform4(fluidRotMat);

			UT_XformOrder rotOrder;
			UT_Vector3D scale, trans;
			fluidRotMat.explode(rotOrder, fluidRot, scale, trans);
			fluidRot.radToDeg();
			fluidPos = volume->getVertex().getPt()->getPos();
		}

		float sizeX = currSolver->fluidSize.x*0.5;
		float sizeY = currSolver->fluidSize.y*0.5;

		int newResX = currSolver->res.x;
		int newResY = currSolver->res.y;


		if(displayX != newResX || displayY != newResY) {
			displayX = newResX;
			displayY = newResY;
			initPixelBuffer(true);
		}
					
		cu::cutilSafeCall(cu::cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		cu::float4 *d_output;
		size_t num_bytes; 
		cu::cutilSafeCall(cu::cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  cuda_pbo_resource));
		cu::cudaMemset(d_output, 0, displayX*displayY*sizeof(cu::float4));

		currSolver->renderFluid(d_output);

		cu::cutilSafeCall(cu::cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

		//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glBindTexture(GL_TEXTURE_2D, gl_Tex);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, displayX, displayY, GL_RGBA, GL_FLOAT, 0);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		glEnable(GL_TEXTURE_2D);

		glPushMatrix();
		glTranslatef(fluidPos.x(),fluidPos.y(),fluidPos.z());
		glRotatef(fluidRot.z(),0,0,1);
		glRotatef(fluidRot.y(),0,1,0);
		glRotatef(fluidRot.x(),1,0,0);

		//glColor3f(1.0,1.0,1.0);
		//glDisable(GL_BLEND);
		//glDisable(GL_LIGHTING);

		//glBegin( GL_QUADS );
		//glTexCoord2f(0.0f, 0.0f); glVertex3f(-sizeX,-sizeY,0.0f);
		//glTexCoord2f(0.0f, 1.0f); glVertex3f(-sizeX,sizeY,0.0f);
		//glTexCoord2f(1.0f, 1.0f); glVertex3f(sizeX,sizeY,0.0f);
		//glTexCoord2f(1.0f, 0.0f); glVertex3f(sizeX,-sizeY,0.0f);
		//glEnd();

		ren.setColor(1,1,1,1);
		ren.blend(0);
		ren.toggleLighting(0);


		const float t0[] = {0,0};
		const float t1[] = {0,1};
		const float t2[] = {1,1};
		const float t3[] = {1,0};

		ren.beginQuads();
		ren.t2DW(t0); ren.vertex3DW(-sizeX,-sizeY,0.0f);
		ren.t2DW(t1); ren.vertex3DW(-sizeX,sizeY,0.0f);
		ren.t2DW(t2); ren.vertex3DW(sizeX,sizeY,0.0f);
		ren.t2DW(t3); ren.vertex3DW(sizeX,-sizeY,0.0f);
		ren.endQuads();

		glDisable(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, 0);

		ren.beginClosedLine();
		ren.vertex3DW(-sizeX,-sizeY,0.0f);
		ren.vertex3DW(-sizeX,sizeY,0.0f);
		ren.vertex3DW(sizeX,sizeY,0.0f);
		ren.vertex3DW(sizeX,-sizeY,0.0f);
		ren.endClosedLine();

		ren.toggleLighting(1);

		glPopMatrix();

	}

	if (fluid3DAh.getI() == 1 || fluid3DSliceAh.getI()== 1) {

		ren.toggleLighting(0);

		VHFluidSolver3D* curr3DSolver = VHFluidSolver3D::solverList[fluidIdAh.getI()];

		UT_Vector4 fluidPos(0,0,0);
		UT_Vector3D fluidRot(0,0,0);

		if(gdp->volumeCount() == 1) {
			GEO_Primitive* pprim = gdp->primitives().head();
			GU_PrimVolume* volume = (GU_PrimVolume *)pprim;
			UT_Matrix4 fluidRotMat;
			volume->getTransform4(fluidRotMat);

			UT_XformOrder rotOrder;
			UT_Vector3D scale, trans;
			fluidRotMat.explode(rotOrder, fluidRot, scale, trans);
			fluidRot.radToDeg();
			fluidPos = volume->getVertex().getPt()->getPos();
		}


		float sizeX = curr3DSolver->fluidSize.x*0.5;
		float sizeY = curr3DSolver->fluidSize.y*0.5;
		float sizeZ = curr3DSolver->fluidSize.z*0.5;

		if(curr3DSolver->drawCube) {
			glPushMatrix();
			glTranslatef(fluidPos.x(),fluidPos.y(),fluidPos.z());
			glRotatef(fluidRot.z(),0,0,1);
			glRotatef(fluidRot.y(),0,1,0);
			glRotatef(fluidRot.x(),1,0,0);
			drawWireCube(sizeX,sizeY,sizeZ, ren);
			glPopMatrix();
		}

			if (fluid3DAh.getI()== 1) {

				curr3DSolver->drawFluid();


			}

			if (fluid3DSliceAh.getI()== 1) {

				curr3DSolver->drawFluidSlice();


			}

			ren.toggleLighting(1);

		}



}

void GR_Fluid::renderShaded(GU_Detail *gdp, RE_Render &ren, const GR_AttribOffset &ptinfo,
		    const GR_DisplayOption *dopt, float lod, const GU_PrimGroupClosure *hidden_geometry)
{
    // We use the wire render again...
    renderWire(gdp, ren, ptinfo, dopt, lod, hidden_geometry);
}

void GR_Fluid::drawWireCube(float x, float y, float z, RE_Render &ren) {

	ren.beginClosedLine();									// Draw A Quad
		ren.vertex3DW( x, y,-z);					// Top Right Of The Quad (Top)
		ren.vertex3DW(-x, y,-z);					// Top Left Of The Quad (Top)
		ren.vertex3DW(-x, y, z);					// Bottom Left Of The Quad (Top)
		ren.vertex3DW( x, y, z);					// Bottom Right Of The Quad (Top)
		//ren.vertex3DW( x, y,-z);
	ren.endClosedLine();
	
	ren.beginClosedLine();
		ren.vertex3DW( x,-y, z);					// Top Right Of The Quad (Bottom)
		ren.vertex3DW(-x,-y, z);					// Top Left Of The Quad (Bottom)
		ren.vertex3DW(-x,-y,-z);					// Bottom Left Of The Quad (Bottom)
		ren.vertex3DW( x,-y,-z);					// Bottom Right Of The Quad (Bottom)
		//glVertex3f( x,-y, z);
	ren.endClosedLine();
		
	ren.beginLines();
		ren.vertex3DW(-x, y, z);					// Top Left Of The Quad (Front)
		ren.vertex3DW(-x,-y, z);					// Bottom Left Of The Quad (Front)
		ren.vertex3DW( x, y, z);					// Top Right Of The Quad (Front)
		ren.vertex3DW( x,-y, z);					// Bottom Right Of The Quad (Front)
	
		ren.vertex3DW(-x,-y,-z);					// Top Left Of The Quad (Back)
		ren.vertex3DW(-x, y,-z);					// Bottom Left Of The Quad (Back)
		ren.vertex3DW( x, y,-z);					// Bottom Right Of The Quad (Back)
		ren.vertex3DW( x,-y,-z);					// Top Right Of The Quad (Back)
	ren.endLines();
}
