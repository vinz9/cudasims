#ifndef FLUIDNODE2D_H
#define FLUIDENODE2D_H

#include <GL/glew.h>

#include <maya/MPxLocatorNode.h> 
#include <maya/MMatrix.h>
#include <maya/MPxNode.h>
#include <maya/MIOStream.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MArrayDataBuilder.h>
#include <maya/MFnMessageAttribute.h>
#include <maya/MPlugArray.h>
#include <maya/MFnTransform.h>
#include <maya/MTime.h>
#include <maya/MVector.h>
#include <maya/MFloatVector.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MDoubleArray.h>
#include <maya/M3dView.h>

#include <maya/MGlobal.h>
#include <maya/MSelectionList.h>
#include <maya/MFnFluid.h>

#include <maya/MDagPath.h>
#include <maya/MFnDagNode.h>

#include <maya/MImage.h>


#include <string>
#include <math.h>

#include "../../CudaCommon/CudaFluidSolver2D/vhFluidSolver.h"


namespace cu{
	#include <cuda_runtime_api.h>
	#include <cuda_gl_interop.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>
}

class fluidNode2D : public MPxLocatorNode
{
public:
						fluidNode2D();
	virtual				~fluidNode2D(); 

	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );



	static	void*		creator();
	static	MStatus		initialize();
	//void postConstructor();
 
	static	MObject		aEmitters;
	static	MObject		aColliders;

	static	MObject		aInTime;

	static MObject		aMayaFluid;

	static	MObject		aOutTime;

	static	MObject		aStartFrame;
	static	MObject		aSubsteps;
	static	MObject		aJacIter;

	static	MObject		aFluidSize;
	static	MObject		aSizeX;
	static	MObject		aSizeY;

	static	MObject		aRes;
	static	MObject		aResX;
	static	MObject		aResY;

	static	MObject		aBorderNegX;
	static	MObject		aBorderPosX;
	static	MObject		aBorderNegY;
	static	MObject		aBorderPosY;
	
	static	MObject		aDensDis;
	static	MObject		aDensBuoyStrength;
	static	MObject		aDensBuoyDir;
	static	MObject		aDensBuoyDirX;
	static	MObject		aDensBuoyDirY;

	static	MObject		aVelDamp;
	static	MObject		aVortConf;

	static	MObject		aNoiseStr;
	static	MObject		aNoiseFreq;
	static	MObject		aNoiseOct;
	static	MObject		aNoiseLacun;
	static	MObject		aNoiseSpeed;
	static	MObject		aNoiseAmp;

	static	MObject		aPreview;
	static	MObject		aPreviewType;
	static	MObject		aMaxBounds;


	static	MTypeId		id;

	bool fluidInitialized;

	MString currentMayaFluidName;
	MFnFluid fluidFn;
	float solverTime;

	unsigned int mResX, mResY, mResZ;

	MStatus updateFluidName(MString newFluidName);

	VHFluidSolver*   fluidSolver;

	virtual void draw( M3dView & view, const MDagPath & path, 
						M3dView::DisplayStyle style,
						M3dView::DisplayStatus status );



};

#endif