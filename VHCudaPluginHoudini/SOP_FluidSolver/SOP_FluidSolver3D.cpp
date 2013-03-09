#pragma warning(disable : 4018)
#pragma warning(disable : 4244)
#pragma warning(disable : 4996)
#pragma warning(disable : 4800)
#pragma warning(disable : 4312)

#include "SOP_FluidSolver3D.h"

static PRM_Default	sizeDefaults[] = { PRM_Default(10.0),	PRM_Default(10.0) , PRM_Default(10.0)};
static PRM_Default	resDefaults[] = { PRM_Default(60),	PRM_Default(60) , PRM_Default(60)};
static PRM_Default	jacIterDefault(30);
static PRM_Default	vortConfDefault(2.0);
static PRM_Default	noiseOctDefault(3);
static PRM_Default	noiseLacunDefault(4.0);
static PRM_Default	noiseSpeedDefault(0.01);
static PRM_Default	noiseAmpDefault(0.5);
static PRM_Default	opaScaleDefault(0.2);
static PRM_Default	lightPosDefaults[] = { PRM_Default(10),	PRM_Default(10) , PRM_Default(10)};
static PRM_Default	shadowStepMulDefault(4);
static PRM_Default	shadowThresDefault(0.9);


static PRM_Range    minOneRange(PRM_RANGE_RESTRICTED,1,PRM_RANGE_FREE,10);
static PRM_Range    minZeroRange(PRM_RANGE_RESTRICTED,0,PRM_RANGE_FREE,10);
static PRM_Range    zeroToOneRange(PRM_RANGE_RESTRICTED,0,PRM_RANGE_RESTRICTED,1);
static PRM_Range    jacIterRange(PRM_RANGE_RESTRICTED,1,PRM_RANGE_FREE,100);


static PRM_Name     switcherName("fluidSolverSwitcher");
static PRM_Default  switcherList[] = {
	PRM_Default(5, "Solver"),
    PRM_Default(3, "Density"),
    PRM_Default(8, "Velocity"),
	PRM_Default(15, "Display"),
	/*PRM_Default(1, "Emitters"),
	PRM_Default(0, "Colliders"),*/
};


/*static PRM_Name eNames[] = {
    PRM_Name("emit#pos",   "Emitter # Pos"),
    PRM_Name("emit#rad", "Radius"),
	PRM_Name("emit#amount", "Amount"),
};

static PRM_Template emitterTemplate[] =
{
    PRM_Template(PRM_XYZ, 3, &eNames[0], PRMzeroDefaults),
	PRM_Template(PRM_FLT, 1, &eNames[1], PRMoneDefaults),
    PRM_Template(PRM_FLT, 1, &eNames[2], PRMoneDefaults),
    PRM_Template()
};*/

static PRM_Name         sliceType[] =
{
    PRM_Name("density", "Density"),
    PRM_Name("velocity", "Velocity"),
    PRM_Name("noise", "Noise"),
	PRM_Name("pressure", "Pressure"),
	PRM_Name("vorticity", "Vorticity"),
	PRM_Name("obstacles", "Obstacles"),

    PRM_Name(0)
};
static PRM_ChoiceList   sliceTypeMenu(PRM_CHOICELIST_SINGLE, sliceType);

static PRM_Name         resDis[] =
{
    PRM_Name("r128", "128"),
    PRM_Name("r256", "256"),
    PRM_Name("r512", "512"),
	/*PRM_Name("r768", "768"),
	PRM_Name("r1024", "1024"),*/

    PRM_Name(0)
};
static PRM_ChoiceList   resDisMenu(PRM_CHOICELIST_SINGLE, resDis);

// The names here have to match the inline evaluation functions
static PRM_Name        names[] = {

	PRM_Name("pos",	"Position"),
	PRM_Name("rot",	"Rotation"),

	PRM_Name("res",	"Resolution"),
	PRM_Name("size", "Size"),

	PRM_Name("borderNegX", "Border -X"),
	PRM_Name("borderPosX", "Border +X"),
	PRM_Name("borderNegY", "Border -Y"),
	PRM_Name("borderPosY", "Border +Y"),
	PRM_Name("borderNegZ", "Border -Z"),
	PRM_Name("borderPosZ", "Border +Z"),

	PRM_Name("substeps", "SubSteps"),
	PRM_Name("jacIter", "Jacobi Iterations"),
	PRM_Name("startFrame", "Start Frame"),

	PRM_Name("densDis", "Density Dissipation"),
	PRM_Name("densBuoyStrength", "Buoyancy Strength"),
	PRM_Name("densBuoyDir", "Buoyancy Dir"),

	PRM_Name("velDamp", "Velocity Damping"),
	PRM_Name("vortConf", "Vorticity Confinement"),
	PRM_Name("noiseStr", "Noise Strength"),
	PRM_Name("noiseFreq", "Noise Frequency"),
	PRM_Name("noiseOct", "Noise Octaves"),
	PRM_Name("noiseLacun", "Noise Lacunarity"),
	PRM_Name("noiseSpeed", "Noise Speed"),
	PRM_Name("noiseAmp", "Noise Amplitude"),

	PRM_Name("preview", "Preview"),
	PRM_Name("drawCube", "Draw Wire Cube"),
	PRM_Name("opaScale", "Opacity Scale"),
	PRM_Name("stepMul", "Step Mul"),
	PRM_Name("displayRes", "Display Res"),

	PRM_Name("doShadows", "Self Shadows"),
	PRM_Name("lightPos", "Light Position"),
	PRM_Name("shadowDens", "Shadow Density"),
	PRM_Name("shadowStepMul", "Shadow Step Mul"),
	PRM_Name("shadowThres", "Shadow Threshold"),

	PRM_Name("displaySlice", "DisplaySlice"),
	PRM_Name("sliceType", "Slice Type"),
	PRM_Name("sliceAxis", "Slice Axis"),
	PRM_Name("slicePos", "Slice Position"),
	PRM_Name("sliceBounds", "Bounds"),

	PRM_Name("copyVel", "Copy Velocity"),
	PRM_Name("copyDens", "Copy Dens"),



};

PRM_Template SOP_FluidSolver3D::myTemplateList[] = {
	
	PRM_Template(PRM_XYZ, 3, &names[0], PRMzeroDefaults),						//pos
	PRM_Template(PRM_XYZ, 3, &names[1], PRMzeroDefaults),						//rot

	PRM_Template(PRM_XYZ, 3, &names[2], resDefaults),							//res
	PRM_Template(PRM_INT_XYZ, 3, &names[3], sizeDefaults),						//size

	PRM_Template(PRM_TOGGLE | PRM_TYPE_JOIN_NEXT, 1, &names[4], PRMoneDefaults),	//borderNegX
	PRM_Template(PRM_TOGGLE, 1, &names[5], PRMoneDefaults),							//borderPosX
	PRM_Template(PRM_TOGGLE | PRM_TYPE_JOIN_NEXT, 1, &names[6], PRMoneDefaults),	//borderNegY
	PRM_Template(PRM_TOGGLE, 1, &names[7], PRMoneDefaults),							//borderPosY
	PRM_Template(PRM_TOGGLE | PRM_TYPE_JOIN_NEXT, 1, &names[8], PRMoneDefaults),	//borderNegZ
	PRM_Template(PRM_TOGGLE, 1, &names[9], PRMoneDefaults),							//borderPosZ

	PRM_Template(PRM_SWITCHER, sizeof(switcherList)/sizeof(PRM_Default), &switcherName, switcherList),

	PRM_Template(PRM_INT, 1, &names[10], PRMoneDefaults, 0, &minOneRange),			//substeps
	PRM_Template(PRM_INT, 1, &names[11], &jacIterDefault, 0, &jacIterRange),			//jacIter
    PRM_Template(PRM_INT,	1, &names[12], PRMoneDefaults),								//startFrame
	PRM_Template(PRM_TOGGLE, 1, &names[40], PRMoneDefaults),							//copyDens
	PRM_Template(PRM_TOGGLE, 1, &names[39], PRMzeroDefaults),							//copyVel

	PRM_Template(PRM_FLT,	1, &names[13], PRMzeroDefaults, 0, &minZeroRange),			//densDis
	PRM_Template(PRM_FLT,	1, &names[14], PRMoneDefaults, 0, &minZeroRange),			//densBuoyStrength
	PRM_Template(PRM_XYZ, 3, &names[15], PRMyaxisDefaults),								//densBuoyDir
	
	PRM_Template(PRM_FLT,	1, &names[16], PRMzeroDefaults, 0, &minZeroRange),			//velDamp
	PRM_Template(PRM_FLT,	1, &names[17], &vortConfDefault, 0, &minZeroRange),			//vortConf

	PRM_Template(PRM_FLT,	1, &names[18], PRMzeroDefaults),							//noiseStr
	PRM_Template(PRM_FLT,	1, &names[19], PRMoneDefaults, 0, &minZeroRange),			//noiseFreq
	PRM_Template(PRM_INT,	1, &names[20], &noiseOctDefault, 0, &minZeroRange),			//noiseOct
	PRM_Template(PRM_FLT,	1, &names[21], &noiseLacunDefault, 0, &minZeroRange),		//noiseLacun
	PRM_Template(PRM_FLT,	1, &names[22], &noiseSpeedDefault),							//noiseSpeed
	PRM_Template(PRM_FLT,	1, &names[23], &noiseAmpDefault, 0, &minZeroRange),			//noiseAmp

	PRM_Template(PRM_TOGGLE, 1, &names[24], PRMzeroDefaults),							//preview
	PRM_Template(PRM_TOGGLE, 1, &names[25], PRMoneDefaults),							//drawCube
	PRM_Template(PRM_FLT,	1, &names[26], PRMoneDefaults, 0, &minZeroRange),			//opaScale
	PRM_Template(PRM_FLT,	1, &names[27], PRMoneDefaults, 0, &minOneRange),			//stepMul
	PRM_Template(PRM_ORD,	1, &names[28], PRMoneDefaults, &resDisMenu),				//displayRes

	PRM_Template(PRM_TOGGLE,	1, &names[29], PRMzeroDefaults),						//doShadows
	PRM_Template(PRM_XYZ,	3, &names[30], lightPosDefaults),							//lightPos
	PRM_Template(PRM_FLT,	1, &names[31], PRMoneDefaults, 0, &minZeroRange),			//shadowDens
	PRM_Template(PRM_FLT,	1, &names[32], &shadowStepMulDefault, 0, &minOneRange),		//shadowStepMul
	PRM_Template(PRM_FLT,	1, &names[33], &shadowThresDefault, 0, &minZeroRange),		//shadowThreshold

	PRM_Template(PRM_TOGGLE,	1, &names[34], PRMzeroDefaults),						//displaySlice
	PRM_Template(PRM_ORD, 1, &names[35], 0, &sliceTypeMenu),							//sliceType
	PRM_Template(PRM_ORD, 1, &names[36], PRMtwoDefaults, &PRMaxisMenu),					//sliceAxis
	PRM_Template(PRM_FLT,	1, &names[37], PRMpointFiveDefaults,0, &zeroToOneRange),		//slicePos
	PRM_Template(PRM_FLT,	1, &names[38], PRMoneDefaults, 0, &minZeroRange),			//sliceBounds


    PRM_Template(),
};


OP_Node * SOP_FluidSolver3D::myConstructor(OP_Network *net, const char *name, OP_Operator *op) {
    return new SOP_FluidSolver3D(net, name, op);
}

SOP_FluidSolver3D::SOP_FluidSolver3D(OP_Network *net, const char *name, OP_Operator *op) : SOP_Node(net, name, op) {

	fluidSolver = new VHFluidSolver3DHoudini();

	f = -1;
	oldf = -1;

}

SOP_FluidSolver3D::~SOP_FluidSolver3D() {

	size_t free, total;

	//cu::cudaThreadExit();

	cu::cudaMemGetInfo(&free, &total);
        
    //printf("mem = %lu %lu\n", free, total);

	delete fluidSolver;

}


OP_ERROR SOP_FluidSolver3D::cookMySop(OP_Context &context) {

	oldf = f;
	f =	context.getFrame();

	double t = context.getTime();

	fluidSolver->fps = OPgetDirector()->getChannelManager()->getSamplesPerSec();

	UT_Interrupt	*boss;
	GU_PrimVolume	*volume;
	GU_PrimVolume	*velXVolume;
	GU_PrimVolume	*velYVolume;
	GU_PrimVolume	*velZVolume;

	OP_Node::flags().timeDep = 1;

	int newResX = RESX(t);
	int newResY = RESY(t);
	int newResZ = RESZ(t);

	if ( newResX != fluidSolver->res.width || newResY != fluidSolver->res.height || newResZ != fluidSolver->res.depth) {
		fluidSolver->changeFluidRes(newResX,newResY,newResZ);

	}

	UT_Vector3 fluidPos(POSX(t), POSY(t), POSZ(t));
	UT_Vector3 fluidRot(ROTX(t), ROTY(t), ROTZ(t));
	fluidRot.degToRad();

	fluidSolver->fluidSize.x = FLUIDSIZEX(t);
	fluidSolver->fluidSize.y = FLUIDSIZEY(t);
	fluidSolver->fluidSize.z = FLUIDSIZEZ(t);

	fluidSolver->borderNegX = BORDERNEGX(t);
	fluidSolver->borderPosX = BORDERPOSX(t);
	fluidSolver->borderNegY = BORDERNEGY(t);
	fluidSolver->borderPosY = BORDERPOSY(t);
	fluidSolver->borderNegZ = BORDERNEGZ(t);
	fluidSolver->borderPosZ = BORDERPOSZ(t);

	fluidSolver->substeps = SUBSTEPS(t);
	fluidSolver->jacIter = JACITER(t);

	fluidSolver->densDis = DENSDIS(t);
	fluidSolver->densBuoyStrength = DENSBUOYSTRENGTH(t);
	float ddirX = DENSBUOYDIRX(t);
	float ddirY = DENSBUOYDIRY(t);
	float ddirZ = DENSBUOYDIRZ(t);
	fluidSolver->densBuoyDir = cu::make_float3(ddirX,ddirY,ddirZ);

	fluidSolver->velDamp = VELDAMP(t);
	fluidSolver->vortConf = VORTCONF(t);

	fluidSolver->noiseStr = NOISESTR(t);
	fluidSolver->noiseFreq = NOISEFREQ(t);
	fluidSolver->noiseOct = NOISEOCT(t);
	fluidSolver->noiseLacun = NOISELACUN(t);
	fluidSolver->noiseSpeed = NOISESPEED(t);
	fluidSolver->noiseAmp = NOISEAMP(t);

	fluidSolver->preview = PREVIEW(t);
	fluidSolver->drawCube = DRAWCUBE(t);
	fluidSolver->opaScale = OPASCALE(t);
	fluidSolver->stepMul = STEPMUL(t);
	fluidSolver->displayRes = DISPLAYRES(t);


	fluidSolver->doShadows = DOSHADOWS(t);
	float lightPosX = LIGHTPOSX(t);
	float lightPosY = LIGHTPOSY(t);
	float lightPosZ = LIGHTPOSZ(t);
	fluidSolver->lightPos = cu::make_float3(lightPosX,lightPosY,lightPosZ);
	fluidSolver->shadowDens = SHADOWDENS(t);
	fluidSolver->shadowStepMul = SHADOWSTEPMUL(t);
	fluidSolver->shadowThres = SHADOWTHRES(t);

	fluidSolver->displaySlice = DISPLAYSLICE(t);
	fluidSolver->sliceType = SLICETYPE(t);
	fluidSolver->sliceAxis = SLICEAXIS(t);
	fluidSolver->slicePos = SLICEPOS(t);
	fluidSolver->sliceBounds = SLICEBOUNDS(t);


    if (error() < UT_ERROR_ABORT) {
		boss = UTgetInterrupt();

	gdp->clearAndDestroy();		

		// Start the interrupt server
		if (boss->opStart("Building Volume")){

			static float		 zero = 0.0;
#ifdef HOUDINI_11
			GB_AttributeRef fluidAtt = gdp->addAttrib("cudaFluid3DPreview", sizeof(int), GB_ATTRIB_INT, &zero);
			gdp->attribs().getElement().setValue<int>(fluidAtt, fluidSolver->preview);

			GB_AttributeRef fluidSliceAtt = gdp->addAttrib("sliceDisplay", sizeof(int), GB_ATTRIB_INT, &zero);
			gdp->attribs().getElement().setValue<int>(fluidSliceAtt, fluidSolver->displaySlice);

			GB_AttributeRef solverIdAtt = gdp->addAttrib("solverId", sizeof(int), GB_ATTRIB_INT, &zero);
			gdp->attribs().getElement().setValue<int>(solverIdAtt, fluidSolver->id);

#else
			GA_WOAttributeRef fluidAtt = gdp->addIntTuple(GA_ATTRIB_DETAIL, "cudaFluid3DPreview", 1);
			gdp->element().setValue<int>(fluidAtt, fluidSolver->preview);
			

			GA_WOAttributeRef fluidSliceAtt = gdp->addIntTuple(GA_ATTRIB_DETAIL, "sliceDisplay", 1);
			gdp->element().setValue<int>(fluidSliceAtt, fluidSolver->displaySlice);

			GA_WOAttributeRef solverIdAtt = gdp->addIntTuple(GA_ATTRIB_DETAIL, "solverId", 1);
			gdp->element().setValue<int>(solverIdAtt, fluidSolver->id);
#endif

			GEO_AttributeHandle         name_gah;
			
			int	def = -1;

#ifdef HOUDINI_11
			gdp->addPrimAttrib("name", sizeof(int), GB_ATTRIB_INDEX, &def);
#else
			gdp->addStringTuple(GA_ATTRIB_PRIMITIVE, "name", 1);
#endif
			name_gah = gdp->getPrimAttribute("name");



			UT_Matrix3              xform;
			const UT_XformOrder volXFormOrder;

			volume = (GU_PrimVolume *)GU_PrimVolume::build(gdp);

#ifdef HOUDINI_11
			volume->getVertex().getPt()->getPos() = fluidPos;
#else
			volume->getVertexElement(0).getPt()->setPos(fluidPos);
#endif

			xform.identity();
			xform.scale(fluidSolver->fluidSize.x*0.5, fluidSolver->fluidSize.y*0.5, fluidSolver->fluidSize.z*0.5);
			xform.rotate(fluidRot.x(), fluidRot.y(), fluidRot.z(), volXFormOrder);

			volume->setTransform(xform);

			name_gah.setElement(volume);
			name_gah.setString("density");

			velXVolume = (GU_PrimVolume *)GU_PrimVolume::build(gdp);
			velYVolume = (GU_PrimVolume *)GU_PrimVolume::build(gdp);
			velZVolume = (GU_PrimVolume *)GU_PrimVolume::build(gdp);

#ifdef HOUDINI_11
			velXVolume->getVertex().getPt()->getPos() = fluidPos;
			velYVolume->getVertex().getPt()->getPos() = fluidPos;
			velZVolume->getVertex().getPt()->getPos() = fluidPos;

#else
			velXVolume->getVertexElement(0).getPt()->setPos(fluidPos);
			velYVolume->getVertexElement(0).getPt()->setPos(fluidPos);
			velZVolume->getVertexElement(0).getPt()->setPos(fluidPos);
#endif
			
			velXVolume->setTransform(xform);
			velYVolume->setTransform(xform);
			velZVolume->setTransform(xform);

			name_gah.setElement(velXVolume);
			name_gah.setString("vel.x");

			name_gah.setElement(velYVolume);
			name_gah.setString("vel.y");

			name_gah.setElement(velZVolume);
			name_gah.setString("vel.z");


			xform.identity();
			xform.rotate(fluidRot.x(), fluidRot.y(), fluidRot.z(), volXFormOrder);
			xform.invert();

			if(lockInputs(context) >= UT_ERROR_ABORT)
				return error();

			if(getInput(0)){
				GU_Detail* emittersInput = (GU_Detail*)inputGeo(0, context);
				GEO_PointList emittersList = emittersInput->points();
				int numEmitters = emittersList.entries();

				if (numEmitters != fluidSolver->nEmit) {
					delete fluidSolver->emitters;
					fluidSolver->nEmit = numEmitters;
					fluidSolver->emitters = new VHFluidEmitter[numEmitters];
				}

				GEO_AttributeHandle radAh, amountAh;
				radAh = emittersInput->getPointAttribute("radius");
				amountAh = emittersInput->getPointAttribute("amount");

				for (int i = 0; i < numEmitters; i++) {

					UT_Vector4 emitPos = emittersList[i]->getPos();
					UT_Vector3 emitPos3(emitPos);

					emitPos3 -= fluidPos;
					emitPos3 = emitPos3*xform;

					fluidSolver->emitters[i].posX = emitPos3.x();
					fluidSolver->emitters[i].posY = emitPos3.y();
					fluidSolver->emitters[i].posZ = emitPos3.z();

					radAh.setElement(emittersList[i]);
					amountAh.setElement(emittersList[i]);

					fluidSolver->emitters[i].radius = radAh.getF(0);
					fluidSolver->emitters[i].amount = amountAh.getF(0);
				}
			} else {

				fluidSolver->nEmit = 0;

			}
		

			if(getInput(1)) {
				GU_Detail* collidersInput = (GU_Detail*)inputGeo(1, context);
		
				GEO_PointList collidersList = collidersInput->points();
				int numColliders = collidersList.entries();

				if (numColliders != fluidSolver->nColliders) {
					delete fluidSolver->colliders;
					fluidSolver->nColliders = numColliders;
					fluidSolver->colliders = new VHFluidCollider[numColliders];
				}

				GEO_AttributeHandle colRadAh;
				colRadAh = collidersInput->getPointAttribute("radius");

				for (int i = 0; i < numColliders; i++) {

					UT_Vector4 colPos = collidersList[i]->getPos();
					UT_Vector3 colPos3(colPos);

					colPos3 -= fluidPos;
					colPos3 = colPos3*xform;

					if (f > STARTFRAME(t)) {
						fluidSolver->colliders[i].oldPosX = fluidSolver->colliders[i].posX;
						fluidSolver->colliders[i].oldPosY = fluidSolver->colliders[i].posY;
						fluidSolver->colliders[i].oldPosZ = fluidSolver->colliders[i].posZ;
					} else {
						fluidSolver->colliders[i].oldPosX = colPos3.x();
						fluidSolver->colliders[i].oldPosY = colPos3.y();
						fluidSolver->colliders[i].oldPosZ = colPos3.z();
					}

					fluidSolver->colliders[i].posX = colPos3.x();
					fluidSolver->colliders[i].posY = colPos3.y();
					fluidSolver->colliders[i].posZ = colPos3.z();

					colRadAh.setElement(collidersList[i]);

					fluidSolver->colliders[i].radius = colRadAh.getF(0);
				}

			} else {
				fluidSolver->nColliders = 0;
			}

			unlockInputs();

			if (f <= STARTFRAME(t)) {

				fluidSolver->resetFluid();

				if (COPYDENS(t)) {

					{
						UT_VoxelArrayWriteHandleF	handle = volume->getVoxelWriteHandle();
						handle->constant(0);

						UT_VoxelArrayWriteHandleF	velXHandle = velXVolume->getVoxelWriteHandle();
						velXHandle->constant(0);
						UT_VoxelArrayWriteHandleF	velYHandle = velYVolume->getVoxelWriteHandle();
						velYHandle->constant(0);
						UT_VoxelArrayWriteHandleF	velZHandle = velZVolume->getVoxelWriteHandle();
						velZHandle->constant(0);
					}

				}


			} else {

				if (f!=oldf) {

					fluidSolver->solveFluid();

				}

				if (COPYDENS(t)) {

					cu::cudaMemcpy( fluidSolver->host_dens, fluidSolver->dev_dens,
					fluidSolver->res.width*fluidSolver->res.height*fluidSolver->res.depth*sizeof(float), cu::cudaMemcpyDeviceToHost );

					{
						UT_VoxelArrayWriteHandleF	handle = volume->getVoxelWriteHandle();

						handle->size(fluidSolver->res.width, fluidSolver->res.height, fluidSolver->res.depth);

						for (int i = 0; i < fluidSolver->res.width; i++) {
							for (int j = 0; j < fluidSolver->res.height; j++) {
								for (int k = 0; k < fluidSolver->res.depth; k++) {
									handle->setValue(i, j, k, fluidSolver->host_dens[k*fluidSolver->res.width*fluidSolver->res.height + j*fluidSolver->res.width + i]);
								}
							}
						}
									

					}

					if (COPYVEL(t)) {

					cu::cudaMemcpy( fluidSolver->host_vel, fluidSolver->dev_vel,
						fluidSolver->res.width*fluidSolver->res.height*fluidSolver->res.depth*sizeof(cu::float4), cu::cudaMemcpyDeviceToHost );

					{
						UT_VoxelArrayWriteHandleF	velXHandle = velXVolume->getVoxelWriteHandle();
						velXHandle->size(fluidSolver->res.width, fluidSolver->res.height, fluidSolver->res.depth);
						UT_VoxelArrayWriteHandleF	velYHandle = velYVolume->getVoxelWriteHandle();
						velYHandle->size(fluidSolver->res.width, fluidSolver->res.height, fluidSolver->res.depth);
						UT_VoxelArrayWriteHandleF	velZHandle = velZVolume->getVoxelWriteHandle();
						velZHandle->size(fluidSolver->res.width, fluidSolver->res.height, fluidSolver->res.depth);

						
						for (int i = 0; i < fluidSolver->res.width; i++) {
							for (int j = 0; j < fluidSolver->res.height; j++) {
								for (int k = 0; k < fluidSolver->res.depth; k++) {
									velXHandle->setValue(i, j, k, fluidSolver->host_vel[4*(k*fluidSolver->res.width*fluidSolver->res.height + j*fluidSolver->res.width + i)]);
									velYHandle->setValue(i, j, k, fluidSolver->host_vel[4*(k*fluidSolver->res.width*fluidSolver->res.height + j*fluidSolver->res.width + i)+1]);
									velZHandle->setValue(i, j, k, fluidSolver->host_vel[4*(k*fluidSolver->res.width*fluidSolver->res.height + j*fluidSolver->res.width + i)+2]);
								}
							}
						}
									

					}


					}

				}
			}


		select(GU_SPrimitive);
		}

		// Tell the interrupt server that we've completed. Must do this
		// regardless of what opStart() returns.
		boss->opEnd();
    }

    gdp->notifyCache(GU_CACHE_ALL);

 
    return error();
}