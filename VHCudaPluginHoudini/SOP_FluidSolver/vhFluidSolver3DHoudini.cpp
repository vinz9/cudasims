#include "vhFluidSolver3DHoudini.h"

void VHFluidSolver3DHoudini::lockOpenGLContext(){

	RE_Render* myRender = RE_Render::getRenderContext(2);
	myRender->lockContextForRender();

}

void VHFluidSolver3DHoudini::unlockOpenGLContext(){

	RE_Render* myRender = RE_Render::getRenderContext(2);
	myRender->unlockContextAfterRender();

}

void VHFluidSolver3DHoudini::calculateTransRot(float* modelViewH, cu::float3* trans, cu::float3* rot){

	UT_Matrix4T<float> houModelView(modelViewH[0],  modelViewH[1],  modelViewH[2],  modelViewH[3],
									modelViewH[4],  modelViewH[5],  modelViewH[6],  modelViewH[7],
									modelViewH[8],  modelViewH[9],  modelViewH[10], modelViewH[11],
									modelViewH[12], modelViewH[13], modelViewH[14], modelViewH[15]);

	UT_XformOrder rotOrder;
	UT_Vector3D hrot;
	UT_Vector3D scale;
	UT_Vector3D htrans;

	houModelView.explode(rotOrder, hrot, scale, htrans);

	hrot.radToDeg();

	trans->x = htrans.x();
	trans->y = htrans.y();
	trans->z = htrans.z();

	rot->x = hrot.x();
	rot->y = hrot.y();
	rot->z = hrot.z();
}