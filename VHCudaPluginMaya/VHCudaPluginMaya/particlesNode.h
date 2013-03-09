#ifndef PARTICLESNODE_H
#define PARTICLESNODE_H

#include <GL/glew.h>

#include <maya/MPxLocatorNode.h> 
#include <maya/MMatrix.h>
#include <maya/MPxNode.h>
#include <maya/MIOStream.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnStringData.h>
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
#include <maya/MAngle.h>
#include <maya/MEulerRotation.h>

#include <maya/MGlobal.h>
#include <maya/MSelectionList.h>
#include <maya/MFnFluid.h>


#include <maya/MDagPath.h>
#include <maya/MFnDagNode.h>


#include <string>
#include <math.h>

#include "../../CudaCommon/CudaParticlesSystem/vhParticlesSystem.h"

namespace cu{
	#include <cuda_runtime_api.h>
	#include <driver_functions.h>
	#include <cuda_gl_interop.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>
}

class particlesNode : public MPxLocatorNode
{
public:
						particlesNode();
	virtual				~particlesNode(); 

	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );

	static	void*		creator();
	static	MStatus		initialize();
	//void postConstructor();
 
	static	MObject		aEmitters;
	static MObject		aFluid;

	static	MObject		aInTime;

	static	MObject		aOutTime;

	static	MObject		aMaxParts;

	static	MObject		aStartFrame;
	static	MObject		aSubsteps;

	static	MObject		aLife;
	static	MObject		aLifeVar;

	static	MObject		aVelDamp;

	static	MObject		aGravityStrength;
	static	MObject		aGravityX;
	static	MObject		aGravityY;
	static	MObject		aGravityZ;
	static	MObject		aGravityDir;


	static	MObject		aFluidStrength;

	static	MObject		aNoiseAmp;
	static	MObject		aNoiseOffsetX;
	static	MObject		aNoiseOffsetY;
	static	MObject		aNoiseOffsetZ;
	static	MObject		aNoiseOffset;

	static	MObject		aNoiseFreq;
	static	MObject		aNoiseOct;
	static	MObject		aNoiseLacun;

	static	MObject		aPreview;
	static	MObject		aShadedMode;

	static  MObject		aSpritePath;

	static	MObject		aLightPosX;
	static	MObject		aLightPosY;
	static	MObject		aLightPosZ;
	static	MObject		aLightPos;

	static	MObject		aLightTargetX;
	static	MObject		aLightTargetY;
	static	MObject		aLightTargetZ;
	static	MObject		aLightTarget;

	static	MObject		aLightColorR;
	static	MObject		aLightColorG;
	static	MObject		aLightColorB;
	static	MObject		aLightColor;

	static	MObject		aColorAttenuationR;
	static	MObject		aColorAttenuationG;
	static	MObject		aColorAttenuationB;
	static	MObject		aColorAttenuation;

	static	MObject		aShadowAlpha;

	static	MObject		aOpaScale;
	static	MObject		aPointSize;

	static	MObject		aStartColorR;
	static	MObject		aStartColorG;
	static	MObject		aStartColorB;
	static	MObject		aStartColor;

	static	MObject		aEndColorR;
	static	MObject		aEndColorG;
	static	MObject		aEndColorB;
	static	MObject		aEndColor;



	static	MTypeId		id;

	//bool fluidInitialized;

	MString currentMayaFluidName;
	MFnFluid fluidFn;
	float solverTime;


	MStatus updateFluidName(MString newFluidName);

	VHFluidSolver3D*   fluidSolver;
	VHParticlesSystem*	particlesSystem;

	virtual void draw( M3dView & view, const MDagPath & path, 
						M3dView::DisplayStyle style,
						M3dView::DisplayStatus status );



};

#endif