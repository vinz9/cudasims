#include "fluidNode3D.h"

MTypeId     fluidNode3D::id( 0x800008 );

MObject     fluidNode3D::aEmitters;
MObject     fluidNode3D::aColliders;

MObject     fluidNode3D::aInTime;
MObject     fluidNode3D::aMayaFluid;
MObject     fluidNode3D::aDensCopy;
MObject     fluidNode3D::aSolverId;
MObject     fluidNode3D::aOutTime;

MObject		fluidNode3D::aStartFrame;
MObject		fluidNode3D::aSubsteps;
MObject		fluidNode3D::aJacIter;

MObject		fluidNode3D::aFluidSize;
MObject		fluidNode3D::aSizeX;
MObject		fluidNode3D::aSizeY;
MObject		fluidNode3D::aSizeZ;

MObject		fluidNode3D::aRes;
MObject		fluidNode3D::aResX;
MObject		fluidNode3D::aResY;
MObject		fluidNode3D::aResZ;

MObject		fluidNode3D::aBorderNegX;
MObject		fluidNode3D::aBorderPosX;
MObject		fluidNode3D::aBorderNegY;
MObject		fluidNode3D::aBorderPosY;
MObject		fluidNode3D::aBorderNegZ;
MObject		fluidNode3D::aBorderPosZ;


MObject		fluidNode3D::aDensDis;
MObject		fluidNode3D::aDensBuoyStrength;

MObject		fluidNode3D::aDensBuoyDir;
MObject		fluidNode3D::aDensBuoyDirX;
MObject		fluidNode3D::aDensBuoyDirY;
MObject		fluidNode3D::aDensBuoyDirZ;

MObject		fluidNode3D::aVelDamp;
MObject		fluidNode3D::aVortConf;

MObject		fluidNode3D::aNoiseStr;
MObject		fluidNode3D::aNoiseFreq;
MObject		fluidNode3D::aNoiseOct;
MObject		fluidNode3D::aNoiseLacun;
MObject		fluidNode3D::aNoiseSpeed;
MObject		fluidNode3D::aNoiseAmp;

MObject		fluidNode3D::aPreview;
MObject		fluidNode3D::aDrawCube;
MObject		fluidNode3D::aOpaScale;
MObject		fluidNode3D::aStepMul;
MObject		fluidNode3D::aDisplayRes;

MObject		fluidNode3D::aDoShadows;
MObject		fluidNode3D::aLightPosX;
MObject		fluidNode3D::aLightPosY;
MObject		fluidNode3D::aLightPosZ;
MObject		fluidNode3D::aLightPos;
MObject		fluidNode3D::aShadowDens;
MObject		fluidNode3D::aShadowStepMul;
MObject		fluidNode3D::aShadowThres;

MObject		fluidNode3D::aDisplaySlice;
MObject		fluidNode3D::aSliceType;
MObject		fluidNode3D::aSliceAxis;
MObject		fluidNode3D::aSlicePos;
MObject		fluidNode3D::aSliceBounds;



//
void* fluidNode3D::creator()
{
	return new fluidNode3D();
}

void fluidNode3D::drawWireCube(float x, float y, float z) {

	glBegin(GL_LINE_STRIP);									// Draw A Quad
		glVertex3f( x, y,-z);					// Top Right Of The Quad (Top)
		glVertex3f(-x, y,-z);					// Top Left Of The Quad (Top)
		glVertex3f(-x, y, z);					// Bottom Left Of The Quad (Top)
		glVertex3f( x, y, z);					// Bottom Right Of The Quad (Top)
		glVertex3f( x, y,-z);
	glEnd();
	
	glBegin(GL_LINE_STRIP);
		glVertex3f( x,-y, z);					// Top Right Of The Quad (Bottom)
		glVertex3f(-x,-y, z);					// Top Left Of The Quad (Bottom)
		glVertex3f(-x,-y,-z);					// Bottom Left Of The Quad (Bottom)
		glVertex3f( x,-y,-z);					// Bottom Right Of The Quad (Bottom)
		glVertex3f( x,-y, z);
	glEnd();
		
	glBegin(GL_LINES);
		glVertex3f(-x, y, z);					// Top Left Of The Quad (Front)
		glVertex3f(-x,-y, z);					// Bottom Left Of The Quad (Front)
		glVertex3f( x, y, z);					// Top Right Of The Quad (Front)
		glVertex3f( x,-y, z);					// Bottom Right Of The Quad (Front)
	
		glVertex3f(-x,-y,-z);					// Top Left Of The Quad (Back)
		glVertex3f(-x, y,-z);					// Bottom Left Of The Quad (Back)
		glVertex3f( x, y,-z);					// Bottom Right Of The Quad (Back)
		glVertex3f( x,-y,-z);					// Top Right Of The Quad (Back)
	glEnd();
}

void fluidNode3D::draw( M3dView & view, const MDagPath & path, 
						M3dView::DisplayStyle style,
						M3dView::DisplayStatus status ){



	view.beginGL(); 

	glPushAttrib(GL_CURRENT_BIT|GL_VIEWPORT_BIT|GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	MObject thisNode = thisMObject();

	MPlug prevPlug( thisNode, aPreview );
	prevPlug.getValue(fluidSolver->preview );

	MPlug displaySlicePlug( thisNode, aDisplaySlice);
	displaySlicePlug.getValue(fluidSolver->displaySlice );

	if(fluidSolver->preview) {

		if ( ( style == M3dView::kFlatShaded ) ||  ( style == M3dView::kGouraudShaded ) ) {  

				MPlug displayResPlug( thisNode, aDisplayRes );
				displayResPlug.getValue(fluidSolver->displayEnum);

				MPlug opaScalePlug( thisNode, aOpaScale );
				opaScalePlug.getValue(fluidSolver->opaScale);

				MPlug stepMulPlug( thisNode, aStepMul );
				stepMulPlug.getValue(fluidSolver->stepMul);

				MPlug doShadowsPlug( thisNode, aDoShadows );
				doShadowsPlug.getValue(fluidSolver->doShadows);

				MPlug shadowDensPlug( thisNode, aShadowDens );
				shadowDensPlug.getValue(fluidSolver->shadowDens);

				MPlug shadowStepMulPlug( thisNode, aShadowStepMul );
				shadowStepMulPlug.getValue(fluidSolver->shadowStepMul);

				MPlug shadowThresPlug( thisNode, aShadowThres );
				shadowThresPlug.getValue(fluidSolver->shadowThres);

				MPlug lightPosXPlug( thisNode, aLightPosX );
				float lightPosX;
				lightPosXPlug.getValue(lightPosX);

				MPlug lightPosYPlug( thisNode, aLightPosY );
				float lightPosY;
				lightPosYPlug.getValue(lightPosY);

				MPlug lightPosZPlug( thisNode, aLightPosZ );
				float lightPosZ;
				lightPosZPlug.getValue(lightPosZ);

				MVector lightPos(lightPosX, lightPosY, lightPosZ);

				MFnTransform fnTransform;
				MFnDagNode fnDag;
				MDagPath path;

				fnDag.setObject(this->thisMObject());
				MObject fluidTransform = fnDag.parent(0);
				fnDag.setObject(fluidTransform);
				fnDag.getPath(path);
				fnTransform.setObject(path);
				MTransformationMatrix fluidMatrix = fnTransform.transformation();

				MTransformationMatrix lightMatrix = MTransformationMatrix();
				lightMatrix.setTranslation(lightPos, MSpace::kWorld);

				lightMatrix = lightMatrix.asMatrix()*fluidMatrix.asMatrixInverse();
				MVector pos = MTransformationMatrix(lightMatrix).getTranslation(MSpace::kWorld);

				fluidSolver->lightPos = cu::make_float3(pos.x,pos.y,pos.z);

				MMatrix mayaModelView;
				view.modelViewMatrix(mayaModelView);

				MTransformationMatrix modelViewTrans(mayaModelView);

				double rot[3];
				MTransformationMatrix::RotationOrder rotOrder;
				modelViewTrans.getRotation(rot, rotOrder);

				MAngle xRot(rot[0]);
				MAngle yRot(rot[1]);
				MAngle zRot(rot[2]);		

				MVector mayaTrans = modelViewTrans.getTranslation(MSpace::kWorld);

				fluidSolver->drawFluid(xRot.asDegrees(),yRot.asDegrees(),zRot.asDegrees(),
										mayaTrans.x, mayaTrans.y, mayaTrans.z);

				}			


			}

			
		if(fluidSolver->displaySlice) {

			MPlug slicePosPlug( thisNode, aSlicePos );
			float slicePos;
			slicePosPlug.getValue(slicePos);

			MPlug sliceAxisPlug( thisNode, aSliceAxis );
			int sliceAxis;
			sliceAxisPlug.getValue(sliceAxis);


			if ( ( style == M3dView::kFlatShaded ) ||  ( style == M3dView::kGouraudShaded ) ) { 

				fluidSolver->drawFluidSlice(0,0,0,0,0,0);

			}


		}

	MPlug drawCubePlug( thisNode, aDrawCube );
	drawCubePlug.getValue(fluidSolver->drawCube);

	if (fluidSolver->drawCube)  
		drawWireCube(fluidSolver->fluidSize.x*0.5,fluidSolver->fluidSize.y*0.5,fluidSolver->fluidSize.z*0.5);

	view.endGL();


}

//
MStatus fluidNode3D::initialize()
{

	MFnUnitAttribute	uAttr;
	MFnNumericAttribute nAttr;
	MFnTypedAttribute	tAttr;
	MFnEnumAttribute	eAttr;
	MFnMessageAttribute	mAttr;
	MStatus				stat;

	aEmitters = mAttr.create("emitters","ems",&stat);
	CHECK_MSTATUS(stat);
	mAttr.setArray(true);
	stat = addAttribute(aEmitters);

	aColliders = mAttr.create("colliders","cols",&stat);
	CHECK_MSTATUS(stat);
	mAttr.setArray(true);
	stat = addAttribute(aColliders);

	aInTime =  uAttr.create( "inTime", "t", MFnUnitAttribute::kTime, 0.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aInTime);

	aMayaFluid = mAttr.create("mayaFluid", "mf", &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aMayaFluid);

	aDensCopy = nAttr.create("densCopy", "dcop", MFnNumericData::kInt, 0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aDensCopy);

	aSolverId = nAttr.create("fluidSolverId", "soId", MFnNumericData::kInt, 0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setWritable(false);
	stat = addAttribute(aSolverId);

	aOutTime =  uAttr.create( "outTime", "ot", MFnUnitAttribute::kTime, 0.0, &stat);
	CHECK_MSTATUS(stat);
	uAttr.setWritable(false);
	stat = addAttribute(aOutTime);

	aStartFrame =  nAttr.create( "startFrame", "sf", MFnNumericData::kInt, 1, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aStartFrame);

	aSubsteps = nAttr.create("substeps", "step", MFnNumericData::kInt, 1, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aSubsteps);

	aJacIter = nAttr.create("jacIter", "ji", MFnNumericData::kInt, 30, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aJacIter);

	//maya doesn't like sx sy

	aSizeX = nAttr.create("sizeX", "fsx", MFnNumericData::kFloat, 10.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aSizeX);

	aSizeY = nAttr.create("sizeY", "fsy", MFnNumericData::kFloat, 10.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aSizeY);

	aSizeZ = nAttr.create("sizeZ", "fsz", MFnNumericData::kFloat, 10.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aSizeZ);

	aFluidSize = nAttr.create("fluidSize", "fs", aSizeX, aSizeY, aSizeZ, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aFluidSize);

	//maya doesn't like rx ry

	aResX = nAttr.create("resX", "rsX", MFnNumericData::kInt, 50, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aResX);

	aResY = nAttr.create("resY", "rsy", MFnNumericData::kInt, 50, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aResY);

	aResZ = nAttr.create("resZ", "rsz", MFnNumericData::kInt, 50, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aResZ);

	aRes = nAttr.create("res", "res", aResX, aResY, aResZ, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	nAttr.setDefault(50,50,50);
	stat = addAttribute(aRes);

	aBorderNegX = nAttr.create("borderNegX", "bnX", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderNegX);

	aBorderPosX = nAttr.create("borderPosX", "bpX", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderPosX);

	aBorderNegY = nAttr.create("borderNegY", "bnY", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderNegY);

	aBorderPosY = nAttr.create("borderPosY", "bpY", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderPosY);

	aBorderNegZ = nAttr.create("borderNegZ", "bnZ", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderNegZ);

	aBorderPosZ = nAttr.create("borderPosZ", "bpZ", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderPosZ);

	aDensDis = nAttr.create("densDis", "dd", MFnNumericData::kFloat, 0.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aDensDis);

	aDensBuoyStrength = nAttr.create("densBuoyStr", "dbs", MFnNumericData::kFloat, 1.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aDensBuoyStrength);

	aDensBuoyDirX = nAttr.create("densBuoyDirX", "dbdx", MFnNumericData::kFloat, 0.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aDensBuoyDirX);
	
	aDensBuoyDirY = nAttr.create("densBuoyDirY", "dbdy", MFnNumericData::kFloat, 1.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aDensBuoyDirY);

	aDensBuoyDirZ = nAttr.create("densBuoyDirZ", "dbdz", MFnNumericData::kFloat, 0.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aDensBuoyDirZ);


	aDensBuoyDir = nAttr.create("densBuoyDir", "dbd", aDensBuoyDirX, aDensBuoyDirY, aDensBuoyDirZ, &stat);
	//aDensBuoyDir = nAttr.create("densBuoyDir", "dbd", aDensBuoyDirX, aDensBuoyDirY, MObject::kNullObj, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	nAttr.setDefault(0.0,1.0,0.0);
	stat = addAttribute(aDensBuoyDir);

	aVelDamp = nAttr.create("velDamp", "vd", MFnNumericData::kFloat, 0.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aVelDamp);

	aVortConf = nAttr.create("vortConf", "vc", MFnNumericData::kFloat, 2.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aVortConf);

	aNoiseStr = nAttr.create("noiseStr", "nst", MFnNumericData::kFloat, 0.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aNoiseStr);

	aNoiseFreq = nAttr.create("noiseFreq", "nfr", MFnNumericData::kFloat, 1.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aNoiseFreq);

	aNoiseOct = nAttr.create("noiseOct", "noc", MFnNumericData::kInt, 3.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aNoiseOct);

	aNoiseLacun = nAttr.create("noiseLacun", "nlc", MFnNumericData::kFloat, 4.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aNoiseLacun);

	aNoiseSpeed = nAttr.create("noiseSpeed", "nsp", MFnNumericData::kFloat, 0.01, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aNoiseSpeed);

	aNoiseAmp = nAttr.create("noiseAmp", "nam", MFnNumericData::kFloat, 0.5, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aNoiseAmp);

	aPreview = nAttr.create("preview", "prv", MFnNumericData::kBoolean, 1, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aPreview);

	aDrawCube = nAttr.create("drawCube", "drc", MFnNumericData::kBoolean, 1, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aDrawCube);

	aOpaScale = nAttr.create("opaScale", "opa", MFnNumericData::kFloat, 0.1, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(0.0);
	nAttr.setSoftMax(1.0);
	stat = addAttribute(aOpaScale);

	aStepMul = nAttr.create("stepMul", "smul", MFnNumericData::kFloat, 1.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(1.0);
	stat = addAttribute(aStepMul);

	aDisplayRes = eAttr.create("displayRes", "dres", 1, &stat);
	CHECK_MSTATUS(stat);
	eAttr.addField("128",0);
	eAttr.addField("256",1);
	eAttr.addField("512",2);
	eAttr.addField("768",3);
	eAttr.addField("1024",4);
	stat = addAttribute(aDisplayRes);

	aDoShadows = nAttr.create("doShadows", "dsh", MFnNumericData::kBoolean, 0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aDoShadows);

	aLightPosX = nAttr.create("lightPosX", "lipx", MFnNumericData::kFloat, 10.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aLightPosX);

	aLightPosY = nAttr.create("lightPosY", "lipy", MFnNumericData::kFloat, 10.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aLightPosY);

	aLightPosZ = nAttr.create("lightPosZ", "lipz", MFnNumericData::kFloat, 0.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aLightPosZ);

	aLightPos = nAttr.create("lightPos", "lipos", aLightPosX, aLightPosY, aLightPosZ, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	nAttr.setDefault(10.0,10.0,0.0);
	stat = addAttribute(aLightPos);

	aShadowDens = nAttr.create("shadowDens", "shd", MFnNumericData::kFloat, 0.9, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(0.0);
	nAttr.setSoftMax(1.0);
	stat = addAttribute(aShadowDens);

	aShadowStepMul = nAttr.create("shadowStepMul", "ssm", MFnNumericData::kFloat, 2.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(1.0);
	stat = addAttribute(aShadowStepMul);

	aShadowThres = nAttr.create("shadowThres", "sht", MFnNumericData::kFloat, 0.9, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(0.0);
	nAttr.setMax(1.0);
	stat = addAttribute(aShadowThres);

	aDisplaySlice = nAttr.create("displaySlice", "disl", MFnNumericData::kBoolean, 0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aDisplaySlice);


	aSliceType = eAttr.create("sliceType", "sty", 0, &stat);
	CHECK_MSTATUS(stat);
	eAttr.addField("Density",0);
	eAttr.addField("Velocity",1);
	eAttr.addField("Noise",2);
	eAttr.addField("Pressure",3);
	eAttr.addField("Vorticity",4);
	eAttr.addField("Obstacles",5);
	stat = addAttribute(aSliceType);

	aSliceAxis = eAttr.create("sliceAxis", "sax", 2, &stat);
	CHECK_MSTATUS(stat);
	eAttr.addField("X",0);
	eAttr.addField("Y",1);
	eAttr.addField("Z",2);
	stat = addAttribute(aSliceAxis);

	aSlicePos = nAttr.create("slicePos", "spo", MFnNumericData::kFloat, 0.5, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(0);
	nAttr.setMax(1);
	nAttr.setKeyable(true);
	stat = addAttribute(aSlicePos);

	aSliceBounds = nAttr.create("sliceBounds", "sbo", MFnNumericData::kFloat, 1.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(0);
	//nAttr.setMax(1);
	nAttr.setKeyable(true);
	stat = addAttribute(aSliceBounds);


	stat = attributeAffects(aInTime, aOutTime);

	return MS::kSuccess;
} 


fluidNode3D::fluidNode3D() {

	currentMayaFluidName = "";
	solverTime = 0.0;


	//glewInit();
	//cu::cutilSafeCall(cu::cudaGLSetGLDevice( cu::cutGetMaxGflopsDeviceId() ));
	
	//fluidInitialized = false;

	fluidSolver = new VHFluidSolver3D();

}


fluidNode3D::~fluidNode3D() {

	
	delete fluidSolver;

	//cu::cutilSafeCall(cu::cudaThreadExit());

}


MStatus fluidNode3D::compute (const MPlug& plug, MDataBlock& data) {

	MStatus returnStatus;

	 if(plug == aOutTime) {

		MDataHandle inTimeHandle = data.inputValue (aInTime, &returnStatus);
		CHECK_MSTATUS( returnStatus );

		MTime currentTime(inTimeHandle.asTime());
		currentTime.setUnit(MTime::uiUnit());

		int currentFrame = (int)currentTime.value();

		MDataHandle startFrameHandle = data.inputValue (aStartFrame, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		int startFrame = startFrameHandle.asInt();



		MFnTransform fnTransform;
		MFnDagNode fnDag;
		MDagPath path;
		MPlugArray emittersArray;

		fnDag.setObject(this->thisMObject());
		MObject fluidTransform = fnDag.parent(0);
		fnDag.setObject(fluidTransform);
		fnDag.getPath(path);
		fnTransform.setObject(path);
		//MVector fluidPos = fnTransform.getTranslation(MSpace::kWorld);
		MTransformationMatrix fluidMatrix = fnTransform.transformation();
		
		MPlug emittersPlug(this->thisMObject(),aEmitters);
		int nPlugs = emittersPlug.numElements();
		int conPlugs = emittersPlug.numConnectedElements();

		if (fluidSolver->nEmit != conPlugs){
			fluidSolver->nEmit = conPlugs;
			delete fluidSolver->emitters;
			fluidSolver->emitters = new VHFluidEmitter[conPlugs];
		}

		int k = 0;

		for ( unsigned int j=0; j<nPlugs; j++ ) {

				bool connected = emittersPlug[j].isConnected();

				if(connected) {
		
					emittersPlug[j].connectedTo(emittersArray, true, false);

					MObject emitter = emittersArray[0].node();

					fnDag.setObject(emitter);
					//std::cout << "Emitter " << j << " : "<< fnDag.name() << std::endl;
					fnDag.getPath(path);

					fnTransform.setObject(path);
			
					//MVector pos = fnTransform.getTranslation(MSpace::kWorld);
					//std::cout << "Emitter " << j << " : "<< pos.y << std::endl;

					MTransformationMatrix emitterMatrix = fnTransform.transformation();
					emitterMatrix = emitterMatrix.asMatrix()*fluidMatrix.asMatrixInverse();
					MVector pos = MTransformationMatrix(emitterMatrix).getTranslation(MSpace::kWorld);

					fluidSolver->emitters[k].posX = pos.x;
					fluidSolver->emitters[k].posY = pos.y;
					fluidSolver->emitters[k].posZ = pos.z;

					MPlug densEmitPlug = fnDag.findPlug("fluidDensityEmission",false);
					//double densEmit = densEmitPlug.asDouble();
					fluidSolver->emitters[k].amount = densEmitPlug.asDouble();

					MPlug distEmitPlug = fnDag.findPlug("maxDistance",false);
					//double distEmit = densEmitPlug.asDouble();
					fluidSolver->emitters[k].radius = distEmitPlug.asDouble();
		
					//std::cout << "Emitter " << j << " : "<< densEmit << std::endl;
					k++;
				}
		}

		MPlugArray collidersArray;

		MPlug collidersPlug(this->thisMObject(),aColliders);

		nPlugs = collidersPlug.numElements();
		conPlugs = collidersPlug.numConnectedElements();

		if (fluidSolver->nColliders != conPlugs){
			fluidSolver->nColliders = conPlugs;
			delete fluidSolver->colliders;
			fluidSolver->colliders = new VHFluidCollider[conPlugs];
		}

		k=0;

		for ( unsigned int j=0; j<nPlugs; j++ ) {

			bool connected = collidersPlug[j].isConnected();

				if(connected) {
		
					collidersPlug[j].connectedTo(collidersArray, true, false);

					MObject collider = collidersArray[0].node();

					fnDag.setObject(collider);
					fnDag.getPath(path);

					fnTransform.setObject(path);

					MTransformationMatrix colliderMatrix = fnTransform.transformation();
					colliderMatrix = colliderMatrix.asMatrix()*fluidMatrix.asMatrixInverse();
					MVector pos = MTransformationMatrix(colliderMatrix).getTranslation(MSpace::kWorld);
			
					//MVector pos = fnTransform.getTranslation(MSpace::kWorld);
					//std::cout << "Emitter " << j << " : "<< pos.y << std::endl;

					if (currentFrame > startFrame) {
						fluidSolver->colliders[k].oldPosX = fluidSolver->colliders[k].posX;
						fluidSolver->colliders[k].oldPosY = fluidSolver->colliders[k].posY;
						fluidSolver->colliders[k].oldPosZ = fluidSolver->colliders[k].posZ;
					} else {
						fluidSolver->colliders[k].oldPosX = pos.x;
						fluidSolver->colliders[k].oldPosY = pos.y;
						fluidSolver->colliders[k].oldPosZ = pos.z;
					}

					fluidSolver->colliders[k].posX = pos.x;
					fluidSolver->colliders[k].posY = pos.y;
					fluidSolver->colliders[k].posZ = pos.z;

					MPlug radiusPlug = fnDag.findPlug("radius",false);
					fluidSolver->colliders[k].radius = radiusPlug.asDouble();

					k++;
				}
		}

		MDataHandle resXHandle = data.inputValue (aResX, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		int newResX = resXHandle.asInt();

		MDataHandle resYHandle = data.inputValue (aResY, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		int newResY = resYHandle.asInt();

		MDataHandle resZHandle = data.inputValue (aResZ, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		int newResZ = resZHandle.asInt();

		if (newResX != fluidSolver->res.width || newResY != fluidSolver->res.height || newResZ != fluidSolver->res.depth) {
			fluidSolver->changeFluidRes(newResX, newResY, newResZ);
		}

		MDataHandle inBorderNegXHandle = data.inputValue (aBorderNegX, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver->borderNegX = inBorderNegXHandle.asBool();

		MDataHandle inBorderPosXHandle = data.inputValue (aBorderPosX, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver->borderPosX = inBorderPosXHandle.asBool();

		MDataHandle inBorderNegYHandle = data.inputValue (aBorderNegY, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver->borderNegY = inBorderNegYHandle.asBool();

		MDataHandle inBorderPosYHandle = data.inputValue (aBorderPosY, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver->borderPosY = inBorderPosYHandle.asBool();

		MDataHandle inBorderNegZHandle = data.inputValue (aBorderNegZ, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver->borderNegZ = inBorderNegZHandle.asBool();

		MDataHandle inBorderPosZHandle = data.inputValue (aBorderPosZ, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver->borderPosZ = inBorderPosZHandle.asBool();

		if(MTime::uiUnit() == MTime::kFilm) {
			fluidSolver->fps = 24;
		} else if(MTime::uiUnit() == MTime::kPALFrame) {
			fluidSolver->fps = 25;
		} else if(MTime::uiUnit() == MTime::kNTSCFrame) {
			fluidSolver->fps = 30;
		} else if(MTime::uiUnit() == MTime::kNTSCField) {
			fluidSolver->fps = 60;
		}

		MDataHandle sizeXHandle = data.inputValue (aSizeX, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver->fluidSize.x = sizeXHandle.asFloat();

		MDataHandle sizeYHandle = data.inputValue (aSizeY, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver->fluidSize.y = sizeYHandle.asFloat();

		MDataHandle sizeZHandle = data.inputValue (aSizeZ, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver->fluidSize.z = sizeZHandle.asFloat();


		MPlug mayaFluidPlug(this->thisMObject(),aMayaFluid);

		bool fluidConnected = mayaFluidPlug.isConnected();
		float* fluidDens;

		if (fluidConnected) {
			MPlugArray fluidArray;
			mayaFluidPlug.connectedTo( fluidArray, true, false );
			MObject mayaFluidObject = fluidArray[0].node();
			fluidFn.setObject(mayaFluidObject);
			fluidDens = fluidFn.density();
		}

		MDataHandle densCopyHandle = data.inputValue (aDensCopy, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		int densCopy = densCopyHandle.asInt();

		if (currentFrame <= startFrame) {

			fluidSolver->resetFluid();

			if (densCopy && fluidConnected) {
					memset(fluidDens,0,fluidSolver->domainSize());
					//cu::cudaMemcpy( fluidDens, fluidSolver->dev_dens, fluidSolver->domainSize(), cu::cudaMemcpyDeviceToHost );
			}
		} else {

			MDataHandle substepsHandle = data.inputValue (aSubsteps, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->substeps = substepsHandle.asInt();

			MDataHandle jacIterHandle = data.inputValue (aJacIter, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->jacIter = jacIterHandle.asInt();

			MDataHandle densDisHandle = data.inputValue (aDensDis, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->densDis = densDisHandle.asFloat();

			MDataHandle densBuoyStrHandle = data.inputValue (aDensBuoyStrength, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->densBuoyStrength = densBuoyStrHandle.asFloat();

			MFloatVector newDir;

			MDataHandle densBuoyDirXHandle = data.inputValue (aDensBuoyDirX, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			newDir.x = densBuoyDirXHandle.asFloat();

			MDataHandle densBuoyDirYHandle = data.inputValue (aDensBuoyDirY, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			newDir.y = densBuoyDirYHandle.asFloat();

			MDataHandle densBuoyDirZHandle = data.inputValue (aDensBuoyDirZ, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			newDir.z = densBuoyDirZHandle.asFloat();

			newDir.normalize();
			fluidSolver->densBuoyDir = cu::make_float3(newDir.x,newDir.y,newDir.z);

			MDataHandle velDampHandle = data.inputValue (aVelDamp, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->velDamp = velDampHandle.asFloat();

			MDataHandle vortConfHandle = data.inputValue (aVortConf, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->vortConf = vortConfHandle.asFloat();

			MDataHandle noiseStrHandle = data.inputValue (aNoiseStr, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->noiseStr = noiseStrHandle.asFloat();

			MDataHandle noiseFreqHandle = data.inputValue (aNoiseFreq, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->noiseFreq = noiseFreqHandle.asFloat();

			MDataHandle noiseOctHandle = data.inputValue (aNoiseOct, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->noiseOct = noiseOctHandle.asInt();

			MDataHandle noiseLacunHandle = data.inputValue (aNoiseLacun, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->noiseLacun = noiseLacunHandle.asFloat();

			MDataHandle noiseSpeedHandle = data.inputValue (aNoiseSpeed, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->noiseSpeed = noiseSpeedHandle.asFloat();

			MDataHandle noiseAmpHandle = data.inputValue (aNoiseAmp, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->noiseAmp = noiseAmpHandle.asFloat();

			//float *myVelX, *myVelY, *myVelZ;
			//fluidFn.getVelocity(myVelX, myVelY, myVelZ);

			if (currentFrame == (startFrame+1)) {

				solverTime = 0;
				fluidSolver->resetFluid();
			}

			fluidSolver->solveFluid();

			if (densCopy && fluidConnected) {
					cu::cudaMemcpy( fluidDens, fluidSolver->dev_dens, fluidSolver->domainSize(), cu::cudaMemcpyDeviceToHost );
				}

			//solverTime +=1;


			MDataHandle outTimeHandle = data.outputValue (aOutTime, &returnStatus);
			CHECK_MSTATUS(returnStatus);
			outTimeHandle.set(currentTime);

			MDataHandle solverIdHandle = data.outputValue (aSolverId, &returnStatus);
			CHECK_MSTATUS(returnStatus);
			solverIdHandle.set(fluidSolver->id);

		}
		
		if(fluidConnected)
			fluidFn.updateGrid();

	 } else {

		return MS::kUnknownParameter;
	}

	return MS::kSuccess;

}


MStatus fluidNode3D::updateFluidName(MString newFluidName) {

	MObject fluidObject;
	MSelectionList list;

		//std::cout << "FluidString :" << newFluid << std::endl;
		//std::cout << "OldFluidString :" << currentFluid << std::endl;
	
	if (newFluidName != currentMayaFluidName) {

		std::cout << "FluidString :" << newFluidName << std::endl;

		MGlobal::getSelectionListByName(newFluidName,list);

		//std::cout << "listlength:"  << list.length() << std::endl;

		if (list.length() == 1) {
			list.getDependNode(0,fluidObject);
			fluidFn.setObject(fluidObject);

			currentMayaFluidName = newFluidName;

		/*	fluidFn.getResolution(mResX, mResY, mResZ);

		std::cout << "FluidName :" << fluidFn.name().asChar() << std::endl;
		std::cout << "Res X :"  << mResX << std::endl;
		std::cout << "Res Y :"  << mResY << std::endl;
		std::cout << "Res Z :"  << mResZ << std::endl;

		 changeFluidRes(mResX,mResY);*/

		} else {
			std::cout << "Wrong Fluid" << std::endl;
			return MS::kFailure;
		}


		
	} else if (currentMayaFluidName == "") {
			std::cout << "No Fluid" << std::endl;
			return MS::kFailure;
	}

	return MS::kSuccess;

}