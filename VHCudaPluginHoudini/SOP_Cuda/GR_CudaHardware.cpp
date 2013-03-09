#pragma warning(disable : 4018)
#pragma warning(disable : 4244)
#pragma warning(disable : 4996)
#pragma warning(disable : 4800)
#pragma warning(disable : 4312)

#include <GL/glew.h>

#include <GL/glut.h>
#include <GL/glext.h>

#include "GR_CudaHardware.h"
#include "../SOP_FluidSolver/SOP_FluidSolver2D.h"
#include "../SOP_FluidSolver/SOP_FluidSolver3D.h"
#include "../SOP_CudaParticles/SOP_CudaParticles.h"

#include <RE/RE_OGLFramebuffer.h>


GR_CudaHardware::GR_CudaHardware() {

	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
	  fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
	}

	cu::cutilSafeCall(cu::cudaGLSetGLDevice(cu::cutGetMaxGflopsDeviceId()));
	cu::cudaGraphicsGLRegisterBuffer(NULL, 0, cu::cudaGraphicsMapFlagsWriteDiscard); //does nothing but prevents crash
	cu::cudaGraphicsUnregisterResource(0);

}

GR_CudaHardware::~GR_CudaHardware() {


	cu::cudaThreadExit();

}

int GR_CudaHardware::preview(const GU_Detail *gdp) {

	GEO_AttributeHandle partsAh= gdp->getDetailAttribute("cudaParticlesPreview");
	partsAh.setElement(gdp);

	GEO_AttributeHandle fluid3DAh= gdp->getDetailAttribute("cudaFluid3DPreview");
	fluid3DAh.setElement(gdp);

	GEO_AttributeHandle fluidAh= gdp->getDetailAttribute("cudaFluidPreview");
	fluidAh.setElement(gdp);

	if (partsAh.getI() || fluid3DAh.getI() || fluidAh.getI())
		return 1;
	else
		return 0;

}


void GR_CudaHardware::renderWire(GU_Detail *gdp, RE_Render &ren, const GR_AttribOffset & /*ptinfo*/,
		    const GR_DisplayOption *dopt, float /*lod*/, const GU_PrimGroupClosure * /*hidden_geometry*/)
{

	GEO_AttributeHandle fluidAh= gdp->getDetailAttribute("cudaFluidPreview");
	fluidAh.setElement(gdp);

	if (fluidAh.getI()== 1) {

		GEO_AttributeHandle fluidIdAh= gdp->getDetailAttribute("solverId");
		fluidIdAh.setElement(gdp);

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

		currSolver->drawFluid(fluidRot.x(), fluidRot.y(), fluidRot.z(),
										fluidPos.x(), fluidPos.y(), fluidPos.z());

	}

	GEO_AttributeHandle partsAh= gdp->getDetailAttribute("cudaParticlesPreview");
	partsAh.setElement(gdp);

	if (partsAh.getI()== 1) {

		GEO_AttributeHandle partsIdAh= gdp->getDetailAttribute("systemId");
		partsIdAh.setElement(gdp);

		ren.toggleLighting(0);

		VHParticlesSystem* currSystem = VHParticlesSystem::systemsList[partsIdAh.getI()];

		glClear(GL_COLOR_BUFFER_BIT);

		/*int maxDrawBuffers;
		glGetIntegerv ( GL_MAX_DRAW_BUFFERS_ARB, &maxDrawBuffers );

		int db;
		glGetIntegerv ( GL_DRAW_BUFFER, &db );

		int dbb;
		glGetIntegerv ( GL_DRAW_BUFFER0, &dbb );

		int dba;
		glGetIntegerv ( GL_AUX_BUFFERS, &dba );*/

		//glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 3);

		//boolean IsFramebufferEXT(uint framebuffer);

		glDisable(GL_BLEND);

		currSystem->draw();

		glEnable(GL_BLEND);

		/*glColor3f(1.0, 0.0, 0.0);
		glBegin(GL_LINES);
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(1,0,0);
		glEnd();*/

		/*GLfloat modelView[16];
		glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

		glColor3f(0.0, 0.0, 1.0);
		glBegin(GL_LINES);
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(modelView[8],modelView[9],modelView[10]);
		glEnd();*/

		ren.toggleLighting(1);

	
	}

	GEO_AttributeHandle fluid3DAh= gdp->getDetailAttribute("cudaFluid3DPreview");
	fluid3DAh.setElement(gdp);

	GEO_AttributeHandle fluid3DSliceAh= gdp->getDetailAttribute("sliceDisplay");
	fluid3DSliceAh.setElement(gdp);

	if (fluid3DAh.getI() == 1 || fluid3DSliceAh.getI()== 1) {

		GEO_AttributeHandle fluidIdAh= gdp->getDetailAttribute("solverId");
		fluidIdAh.setElement(gdp);

		ren.toggleLighting(0);

		VHFluidSolver3D* curr3DSolver = VHFluidSolver3D::solverList[fluidIdAh.getI()];

		UT_Vector4 fluidPos(0,0,0);
		UT_Vector3D fluidRot(0,0,0);

		if(gdp->volumeCount() > 0) {
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

				curr3DSolver->drawFluid(fluidRot.x(), fluidRot.y(), fluidRot.z(),
										fluidPos.x(), fluidPos.y(), fluidPos.z());

			}

			if (fluid3DSliceAh.getI()== 1) {

				curr3DSolver->drawFluidSlice(fluidRot.x(), fluidRot.y(), fluidRot.z(),
										fluidPos.x(), fluidPos.y(), fluidPos.z());

			}

			ren.toggleLighting(1);

		}




}


void GR_CudaHardware::renderShaded(GU_Detail *gdp, RE_Render &ren, const GR_AttribOffset &ptinfo,
		    const GR_DisplayOption *dopt, float lod, const GU_PrimGroupClosure *hidden_geometry)
{
    // We use the wire render again...
    renderWire(gdp, ren, ptinfo, dopt, lod, hidden_geometry);
}

void GR_CudaHardware::drawWireCube(float x, float y, float z, RE_Render &ren) {

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
