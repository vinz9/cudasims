#pragma warning(disable : 4018)
#pragma warning(disable : 4244)
#pragma warning(disable : 4996)
#pragma warning(disable : 4800)
#pragma warning(disable : 4312)

#include <GL/glew.h>

#include "SOP_FluidSolver2D.h"

static PRM_Default	sizeDefaults[] = { PRM_Default(10.0),	PRM_Default(10.0) , PRM_Default(1.0)};
static PRM_Default	resDefaults[] = { PRM_Default(150),	PRM_Default(150) , PRM_Default(1)};
static PRM_Default	jacIterDefault(50);
static PRM_Default	vortConfDefault(2.0);
static PRM_Default	noiseOctDefault(3);
static PRM_Default	noiseLacunDefault(4.0);
static PRM_Default	noiseSpeedDefault(0.01);
static PRM_Default	noiseAmpDefault(0.5);

static PRM_Range    minOneRange(PRM_RANGE_RESTRICTED,1,PRM_RANGE_FREE,10);
static PRM_Range    minZeroRange(PRM_RANGE_RESTRICTED,0,PRM_RANGE_FREE,10);
static PRM_Range    zeroToOneRange(PRM_RANGE_RESTRICTED,0,PRM_RANGE_RESTRICTED,1);
static PRM_Range    jacIterRange(PRM_RANGE_RESTRICTED,1,PRM_RANGE_FREE,100);

//static PRM_Range    freeRange(PRM_RANGE_RESTRICTED,1,PRM_RANGE_RESTRICTED,100);

static PRM_Name     switcherName("fluidSolverSwitcher");
static PRM_Default  switcherList[] = {
	PRM_Default(3, "Solver"),
    PRM_Default(3, "Density"),
    PRM_Default(8, "Velocity"),
	PRM_Default(3, "Display"),
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

static PRM_Name         previewChoices[] =
{
    PRM_Name("density", "Density"),
    PRM_Name("velocity", "Velocity"),
    PRM_Name("noise", "Noise"),
	PRM_Name("pressure", "Pressure"),
	PRM_Name("vorticity", "Vorticity"),
	PRM_Name("obstacles", "Obstacles"),

    PRM_Name(0)
};
static PRM_ChoiceList   previewMenu(PRM_CHOICELIST_SINGLE, previewChoices);



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

	PRM_Name("preview", "Preview"),
	PRM_Name("previewType", "Preview Type"),
	PRM_Name("bounds", "Bounds"),

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

	//PRM_Name("emitNum",  "Number of Emitters"),

};

PRM_Template SOP_FluidSolver2D::myTemplateList[] = {
	
	PRM_Template(PRM_XYZ, 3, &names[0], PRMzeroDefaults),		//position
	PRM_Template(PRM_XYZ, 3, &names[1], PRMzeroDefaults),		//rotation

	PRM_Template(PRM_XYZ, 2, &names[2], resDefaults),			//res
	PRM_Template(PRM_INT_XYZ, 3, &names[3], sizeDefaults),		//size

	PRM_Template(PRM_TOGGLE | PRM_TYPE_JOIN_NEXT, 1, &names[4], PRMoneDefaults),	//border-x
	PRM_Template(PRM_TOGGLE, 1, &names[5], PRMoneDefaults),							//border+x
	PRM_Template(PRM_TOGGLE | PRM_TYPE_JOIN_NEXT, 1, &names[6], PRMoneDefaults),	//border-y
	PRM_Template(PRM_TOGGLE, 1, &names[7], PRMoneDefaults),							//border+y

	PRM_Template(PRM_SWITCHER, sizeof(switcherList)/sizeof(PRM_Default), &switcherName, switcherList),

	PRM_Template(PRM_INT, 1, &names[11], PRMoneDefaults, 0, &minOneRange),			//substeps
	PRM_Template(PRM_INT, 1, &names[12], &jacIterDefault,0, &jacIterRange),			//jacIter
    PRM_Template(PRM_INT,	1, &names[13], PRMoneDefaults),							//startframe

	PRM_Template(PRM_FLT,	1, &names[14], PRMzeroDefaults, 0, &minZeroRange),		//densDis
	PRM_Template(PRM_FLT,	1, &names[15], PRMoneDefaults, 0, &minZeroRange),		//buoystr
	PRM_Template(PRM_XYZ, 2, &names[16], PRMyaxisDefaults),							//buoydir
	
	PRM_Template(PRM_FLT,	1, &names[17], PRMzeroDefaults, 0, &minZeroRange),		//veldamp
	PRM_Template(PRM_FLT,	1, &names[18], &vortConfDefault, 0, &minZeroRange),		//vortconf

	PRM_Template(PRM_FLT,	1, &names[19], PRMzeroDefaults),						//noiseStr
	PRM_Template(PRM_FLT,	1, &names[20], PRMoneDefaults, 0, &minZeroRange),		//noiseFreq
	PRM_Template(PRM_INT,	1, &names[21], &noiseOctDefault, 0, &minZeroRange),		//noiseOct
	PRM_Template(PRM_FLT,	1, &names[22], &noiseLacunDefault, 0, &minZeroRange),	//noiseLac
	PRM_Template(PRM_FLT,	1, &names[23], &noiseSpeedDefault),						//noiseSpeed
	PRM_Template(PRM_FLT,	1, &names[24], &noiseAmpDefault),						//noiseamp

	PRM_Template(PRM_TOGGLE, 1, &names[8], PRMzeroDefaults),						//preview
	PRM_Template(PRM_ORD, 1, &names[9], 0, &previewMenu),							//previewType
	PRM_Template(PRM_FLT, 1, &names[10], PRMoneDefaults),							//bounds

	//PRM_Template(PRM_MULTITYPE_LIST, emitterTemplate, 0, &names[20]),

    PRM_Template(),
};

OP_Node * SOP_FluidSolver2D::myConstructor(OP_Network *net, const char *name, OP_Operator *op) {
    return new SOP_FluidSolver2D(net, name, op);
}

SOP_FluidSolver2D::SOP_FluidSolver2D(OP_Network *net, const char *name, OP_Operator *op) : SOP_Node(net, name, op) {

	fluidSolver = new VHFluidSolver2DHoudini();

	f = -1;
	oldf = -1;

}

SOP_FluidSolver2D::~SOP_FluidSolver2D() {

	delete fluidSolver;

}


OP_ERROR SOP_FluidSolver2D::cookMySop(OP_Context &context) {

	oldf = f;

	double t = context.getTime();
	int f =	context.getFrame();
	UT_Interrupt	*boss;
	GU_PrimVolume	*volume;

	OP_Node::flags().timeDep = 1;
	fluidSolver->fps = OPgetDirector()->getChannelManager()->getSamplesPerSec();


	int newResX = RESX(t);
	int newResY = RESY(t);

	if ( newResX != fluidSolver->res.x || newResY != fluidSolver->res.y) {
		fluidSolver->changeFluidRes(newResX,newResY);

	}

	UT_Vector3 fluidPos(POSX(t), POSY(t), POSZ(t));
	UT_Vector3 fluidRot(ROTX(t), ROTY(t), ROTZ(t));
	fluidRot.degToRad();

	fluidSolver->fluidSize.x = FLUIDSIZEX(t);
	fluidSolver->fluidSize.y = FLUIDSIZEY(t);

	fluidSolver->borderNegX = BORDERNEGX(t);
	fluidSolver->borderPosX = BORDERPOSX(t);
	fluidSolver->borderNegY = BORDERNEGY(t);
	fluidSolver->borderPosY = BORDERPOSY(t);

	fluidSolver->preview = PREVIEW(t);
	fluidSolver->previewType = PREVIEWTYPE(t);
	fluidSolver->bounds = BOUNDS(t);

	fluidSolver->substeps = SUBSTEPS(t);
	fluidSolver->jacIter = JACITER(t);

	fluidSolver->densDis = DENSDIS(t);
	fluidSolver->densBuoyStrength = DENSBUOYSTRENGTH(t);
	float ddirX = DENSBUOYDIRX(t);
	float ddirY = DENSBUOYDIRY(t);
	fluidSolver->densBuoyDir = cu::make_float2(ddirX,ddirY);

	fluidSolver->velDamp = VELDAMP(t);
	fluidSolver->vortConf = VORTCONF(t);

	fluidSolver->noiseStr = NOISESTR(t);
	fluidSolver->noiseFreq = NOISEFREQ(t);
	fluidSolver->noiseOct = NOISEOCT(t);
	fluidSolver->noiseLacun = NOISELACUN(t);
	fluidSolver->noiseSpeed = NOISESPEED(t);
	fluidSolver->noiseAmp = NOISEAMP(t);

    if (error() < UT_ERROR_ABORT) {
			boss = UTgetInterrupt();

		gdp->clearAndDestroy();		

		// Start the interrupt server
		if (boss->opStart("Building Volume")){

			static float		 zero = 0.0;

#ifdef HOUDINI_11
			GB_AttributeRef fluidAtt = gdp->addAttrib("cudaFluidPreview", sizeof(int), GB_ATTRIB_INT, &zero);
			gdp->attribs().getElement().setValue<int>(fluidAtt, fluidSolver->preview);

			GB_AttributeRef solverIdAtt = gdp->addAttrib("solverId", sizeof(int), GB_ATTRIB_INT, &zero);
			gdp->attribs().getElement().setValue<int>(solverIdAtt, fluidSolver->id);
#else
			GA_WOAttributeRef fluidAtt = gdp->addIntTuple(GA_ATTRIB_DETAIL, "cudaFluidPreview", 1);
			gdp->element().setValue<int>(fluidAtt, fluidSolver->preview);

			GA_WOAttributeRef solverIdAtt = gdp->addIntTuple(GA_ATTRIB_DETAIL, "solverId", 1);
			gdp->element().setValue<int>(solverIdAtt, fluidSolver->id);
#endif


			UT_Matrix3              xform;
			const UT_XformOrder volXFormOrder;
			
			volume = (GU_PrimVolume *)GU_PrimVolume::build(gdp);

#ifdef HOUDINI_11
			volume->getVertex().getPt()->getPos() = fluidPos;
#else
			volume->getVertexElement(0).getPt()->setPos(fluidPos);
#endif

			xform.identity();
			xform.scale(fluidSolver->fluidSize.x*0.5, fluidSolver->fluidSize.y*0.5, 0.25);
			xform.rotate(fluidRot.x(), fluidRot.y(), fluidRot.z(), volXFormOrder);

			volume->setTransform(xform);
			

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
					fluidSolver->emitters = new FluidEmitter[numEmitters];
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
					fluidSolver->colliders = new Collider[numColliders];
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
					} else {
						fluidSolver->colliders[i].oldPosX = colPos3.x();
						fluidSolver->colliders[i].oldPosY = colPos3.y();
					}

					fluidSolver->colliders[i].posX = colPos3.x();
					fluidSolver->colliders[i].posY = colPos3.y();

					colRadAh.setElement(collidersList[i]);

					fluidSolver->colliders[i].radius = colRadAh.getF(0);
				}

			} else {
				fluidSolver->nColliders = 0;
			}

			unlockInputs();

			if (f <= STARTFRAME(t)) {

				fluidSolver->resetFluid();

				if (fluidSolver->preview != 1) {
					{
						UT_VoxelArrayWriteHandleF	handle = volume->getVoxelWriteHandle();
						handle->constant(0);
					}
				}


			} else {

				if (f!=oldf) {

					fluidSolver->solveFluid();

				}

				if (fluidSolver->preview != 1) {
					
						cu::cudaMemcpy( fluidSolver->host_dens, fluidSolver->dev_dens,
						fluidSolver->res.x*fluidSolver->res.y*sizeof(float), cu::cudaMemcpyDeviceToHost );
				
					{
						UT_VoxelArrayWriteHandleF	handle = volume->getVoxelWriteHandle();

						handle->size(fluidSolver->res.x, fluidSolver->res.y, 1);

						for (int i = 0; i < fluidSolver->res.x; i++) {
							for (int j = 0; j < fluidSolver->res.y; j++) {
								handle->setValue(i, j, 0, fluidSolver->host_dens[(j*fluidSolver->res.x + i)]);
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