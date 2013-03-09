#include "fluidNode2D.h"

MTypeId     fluidNode2D::id( 0x800007 );

MObject     fluidNode2D::aEmitters;
MObject     fluidNode2D::aColliders;

MObject     fluidNode2D::aInTime;
MObject     fluidNode2D::aMayaFluid;
MObject     fluidNode2D::aOutTime;

MObject		fluidNode2D::aStartFrame;
MObject		fluidNode2D::aSubsteps;
MObject		fluidNode2D::aJacIter;

MObject		fluidNode2D::aFluidSize;
MObject		fluidNode2D::aSizeX;
MObject		fluidNode2D::aSizeY;

MObject		fluidNode2D::aRes;
MObject		fluidNode2D::aResX;
MObject		fluidNode2D::aResY;

MObject		fluidNode2D::aBorderNegX;
MObject		fluidNode2D::aBorderPosX;
MObject		fluidNode2D::aBorderNegY;
MObject		fluidNode2D::aBorderPosY;

MObject		fluidNode2D::aDensDis;
MObject		fluidNode2D::aDensBuoyStrength;

MObject		fluidNode2D::aDensBuoyDir;
MObject		fluidNode2D::aDensBuoyDirX;
MObject		fluidNode2D::aDensBuoyDirY;

MObject		fluidNode2D::aVelDamp;
MObject		fluidNode2D::aVortConf;

MObject		fluidNode2D::aNoiseStr;
MObject		fluidNode2D::aNoiseFreq;
MObject		fluidNode2D::aNoiseOct;
MObject		fluidNode2D::aNoiseLacun;
MObject		fluidNode2D::aNoiseSpeed;
MObject		fluidNode2D::aNoiseAmp;

MObject		fluidNode2D::aPreview;
MObject		fluidNode2D::aPreviewType;
MObject		fluidNode2D::aMaxBounds;


//
void* fluidNode2D::creator()
{
	return new fluidNode2D();
}


void fluidNode2D::draw( M3dView & view, const MDagPath & path, 
						M3dView::DisplayStyle style,
						M3dView::DisplayStatus status ){

	view.beginGL(); 

	MObject thisNode = thisMObject();

	MPlug prevPlug( thisNode, aPreview );
	prevPlug.getValue(fluidSolver->preview );


	if(fluidSolver->preview) {

		//if ( ( style == M3dView::kFlatShaded ) ||  ( style == M3dView::kGouraudShaded ) ) {  
			// Push the color settings
			// 
			glPushAttrib( GL_CURRENT_BIT );


			MPlug previewTypePlug( thisNode, aPreviewType );
			previewTypePlug.getValue(fluidSolver->previewType);

			MPlug maxBoundsPlug( thisNode, aMaxBounds );
			maxBoundsPlug.getValue(fluidSolver->bounds);

			fluidSolver->drawFluid(0,0,0,0,0,0);

			glPopAttrib();

		//}

		

	}


	view.endGL();


}

//
MStatus fluidNode2D::initialize()
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

	aJacIter = nAttr.create("jacIter", "ji", MFnNumericData::kInt, 50, &stat);
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

	aFluidSize = nAttr.create("fluidSize", "fs", aSizeX, aSizeY, MObject::kNullObj, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aFluidSize);

	//maya doesn't like rx ry

	aResX = nAttr.create("resX", "rsX", MFnNumericData::kInt, 200, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aResX);

	aResY = nAttr.create("resY", "rsy", MFnNumericData::kInt, 200, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aResY);

	aRes = nAttr.create("res", "res", aResX, aResY, MObject::kNullObj, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aRes);

	aBorderNegX = nAttr.create("borderNegX", "bnX", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderNegX);

	aBorderPosX = nAttr.create("borderPosX", "bpX", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderPosX);

	/*aBorderX = nAttr.create("borderX", "brx", aBorderNegX, aBorderPosX, MObject::kNullObj, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderX);*/

	aBorderNegY = nAttr.create("borderNegY", "bnY", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderNegY);

	aBorderPosY = nAttr.create("borderPosY", "bpY", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderPosY);

	/*aBorderY = nAttr.create("borderY", "bry", aBorderNegY, aBorderPosY, MObject::kNullObj, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderY);*/



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

	aDensBuoyDir = nAttr.create("densBuoyDir", "dbd", aDensBuoyDirX, aDensBuoyDirY, MObject::kNullObj, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
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

	aPreviewType = eAttr.create("previewType", "prty", 0, &stat);
	CHECK_MSTATUS(stat);
	eAttr.addField("Density",0);
	eAttr.addField("Velocity",1);
	eAttr.addField("Noise",2);
	eAttr.addField("Pressure",3);
	eAttr.addField("Vorticity",4);
	eAttr.addField("Obstacles",5);
	stat = addAttribute(aPreviewType);

	aMaxBounds = nAttr.create("maxBounds", "mbo", MFnNumericData::kFloat, 1.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(0);
	//nAttr.setMax(1);
	nAttr.setKeyable(true);
	stat = addAttribute(aMaxBounds);


	stat = attributeAffects(aInTime, aOutTime);

	return MS::kSuccess;
} 

fluidNode2D::fluidNode2D() {

	currentMayaFluidName = "";
	solverTime = 0.0;

	/*GLenum err = glewInit();
	if (GLEW_OK != err)
	{
	  std::cout << glewGetErrorString(err) << std::endl;
	}

	cu::cutilSafeCall(cu::cudaGLSetGLDevice( cu::cutGetMaxGflopsDeviceId() ));*/

	fluidSolver = new VHFluidSolver();

	//fluidInitialized = false;


}


fluidNode2D::~fluidNode2D() {

	delete fluidSolver;

	//cu::cutilSafeCall(cu::cudaThreadExit());

}



MStatus fluidNode2D::compute (const MPlug& plug, MDataBlock& data) {

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
			fluidSolver->emitters = new FluidEmitter[conPlugs];
		}

		int k = 0;

		for ( unsigned int j=0; j<nPlugs; j++ ) {

			bool connected = emittersPlug[j].isConnected();

				if(connected) {
		
					emittersPlug[j].connectedTo(emittersArray, true, false);

					MObject emitter = emittersArray[0].node();

					fnDag.setObject(emitter);
					fnDag.getPath(path);

					fnTransform.setObject(path);
			
					//MVector pos = fnTransform.getTranslation(MSpace::kWorld);
					//std::cout << "Emitter " << j << " : "<< pos.y << std::endl;

					MTransformationMatrix emitterMatrix = fnTransform.transformation();
					emitterMatrix = emitterMatrix.asMatrix()*fluidMatrix.asMatrixInverse();
					MVector pos = MTransformationMatrix(emitterMatrix).getTranslation(MSpace::kWorld);

					fluidSolver->emitters[k].posX = pos.x;
					fluidSolver->emitters[k].posY = pos.y;

					MPlug densEmitPlug = fnDag.findPlug("fluidDensityEmission",false);
					//double densEmit = densEmitPlug.asDouble();
					fluidSolver->emitters[k].amount = densEmitPlug.asDouble();

					if(pos.z != 0)
						fluidSolver->emitters[k].amount = 0;

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
			fluidSolver->colliders = new Collider[conPlugs];

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
					} else {
						fluidSolver->colliders[k].oldPosX = pos.x;
						fluidSolver->colliders[k].oldPosY = pos.y;
					}

					fluidSolver->colliders[k].posX = pos.x;
					fluidSolver->colliders[k].posY = pos.y;

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

		if (newResX != fluidSolver->res.x || newResY != fluidSolver->res.y) {
			fluidSolver->changeFluidRes(newResX, newResY);
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

		if(MTime::uiUnit() == MTime::kFilm) {
			fluidSolver->fps = 24;
		} else if(MTime::uiUnit() == MTime::kPALFrame) {
			fluidSolver->fps = 25;
		} else if(MTime::uiUnit() == MTime::kNTSCFrame) {
			fluidSolver->fps = 30;
		} else if(MTime::uiUnit() == MTime::kNTSCField) {
			fluidSolver->fps = 60;
		}

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

		MDataHandle inPreviewHandle = data.inputValue (aPreview, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver->preview = inPreviewHandle.asInt();

		if (currentFrame <= startFrame) {

			fluidSolver->resetFluid();

			if (fluidSolver->preview == 0 && fluidConnected)
				memset(fluidDens,0,fluidSolver->domainSize());
			
		} else {

			MDataHandle substepsHandle = data.inputValue (aSubsteps, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->substeps = substepsHandle.asInt();

			MDataHandle jacIterHandle = data.inputValue (aJacIter, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->jacIter = jacIterHandle.asInt();

			MDataHandle sizeXHandle = data.inputValue (aSizeX, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->fluidSize.x = sizeXHandle.asFloat();

			MDataHandle sizeYHandle = data.inputValue (aSizeY, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver->fluidSize.y = sizeYHandle.asFloat();

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

			newDir.z = 0.0;
			newDir.normalize();
			fluidSolver->densBuoyDir = cu::make_float2(newDir.x,newDir.y);

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

			if (fluidSolver->preview == 0) {

				if (fluidConnected)
					cu::cudaMemcpy( fluidDens, fluidSolver->dev_dens, fluidSolver->domainSize(), cu::cudaMemcpyDeviceToHost );

			}

			//solverTime +=1;


			MDataHandle outTimeHandle = data.outputValue (aOutTime, &returnStatus);
			CHECK_MSTATUS(returnStatus);

			outTimeHandle.set(currentTime);

		}
		
		if (fluidConnected)
			fluidFn.updateGrid();

	 } else {

		return MS::kUnknownParameter;
	}

	return MS::kSuccess;

}


MStatus fluidNode2D::updateFluidName(MString newFluidName) {

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