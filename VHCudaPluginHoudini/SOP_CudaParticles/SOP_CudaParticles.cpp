#pragma warning(disable : 4018)
#pragma warning(disable : 4244)
#pragma warning(disable : 4996)
#pragma warning(disable : 4800)
#pragma warning(disable : 4312)

#include "SOP_CudaParticles.h"

static PRM_Default	maxPartsDefault(2000000);
static PRM_Default	lifeDefault(4.0);
static PRM_Default	lifeVarDefault(1.0);
static PRM_Default	velDampDefault(0.1);
static PRM_Default	gravityStrDefault(0.0);
static PRM_Default	gravityDirDefaults[] = { PRM_Default(0.0),	PRM_Default(-1.0) , PRM_Default(0.0)};
static PRM_Default	opacityDefault(0.05);
static PRM_Default	startColorDefaults[] = { PRM_Default(0.39),	PRM_Default(0.17) , PRM_Default(0.95)};
static PRM_Default	endColorDefaults[] = { PRM_Default(1.0),	PRM_Default(0.97) , PRM_Default(0.33)};


static PRM_Default	resDefaults[] = { PRM_Default(60),	PRM_Default(60) , PRM_Default(60)};


static PRM_Range    minOneRange(PRM_RANGE_RESTRICTED,1,PRM_RANGE_FREE,10);
static PRM_Range    minZeroRange(PRM_RANGE_RESTRICTED,0,PRM_RANGE_FREE,10);
static PRM_Range    zeroToOneRange(PRM_RANGE_RESTRICTED,0,PRM_RANGE_RESTRICTED,1);
static PRM_Range    pointSizeRange(PRM_RANGE_RESTRICTED,1,PRM_RANGE_FREE,10);

static PRM_Name     switcherName("particlesSystemSwitcher");
static PRM_Default  switcherList[] = {
    PRM_Default(9, "Velocity"),
	PRM_Default(5, "Display"),
};


// The names here have to match the inline evaluation functions
static PRM_Name        names[] = {

	PRM_Name("maxParts",	"Max Particles"),
	PRM_Name("startFrame",	"Start Frame"),
	PRM_Name("substeps",	"Substeps"),
	PRM_Name("life",	"Life"),
	PRM_Name("lifeVar",	"Life Var"),

	PRM_Name("velDamp",		"Velocity Damping"),
	PRM_Name("gravityStr",		"Gravity Strength"),
	PRM_Name("gravityDir",		"Gravity Dir"),
	PRM_Name("fluidStr",		"Fluid Strength"),


	PRM_Name("preview",	"Preview"),
	PRM_Name("pointSize","Point Size"),
	PRM_Name("opacity","Opacity"),
	PRM_Name("startColor","Start Color"),
	PRM_Name("endColor","End Color"),

	PRM_Name("noiseAmp","Noise Amp"),
	PRM_Name("noiseFreq","Noise Freq"),
	PRM_Name("noiseOct","Noise Oct"),
	PRM_Name("noiseLacun","Noise Lac"),
	PRM_Name("noiseOffset","Noise Offset"),




};

PRM_Template SOP_CudaParticles::myTemplateList[] = {
	
	PRM_Template(PRM_INT, 1, &names[0], &maxPartsDefault, 0, &minZeroRange),			//maxParts
	PRM_Template(PRM_INT, 1, &names[1], PRMoneDefaults),								//startframe
	PRM_Template(PRM_INT, 1, &names[2], PRMoneDefaults, 0, &minOneRange),				//substeps
	PRM_Template(PRM_FLT | PRM_TYPE_JOIN_NEXT, 1, &names[3], &lifeDefault, 0, &minZeroRange),		//life
	PRM_Template(PRM_FLT, 1, &names[4], PRMoneDefaults, 0, &minZeroRange),				//lifevar

	PRM_Template(PRM_SWITCHER, sizeof(switcherList)/sizeof(PRM_Default), &switcherName, switcherList),

	PRM_Template(PRM_FLT, 1, &names[5], &velDampDefault, 0, &minZeroRange),				//velDamp
	PRM_Template(PRM_FLT, 1, &names[6], &gravityStrDefault),								//gravityStr
	PRM_Template(PRM_XYZ, 3, &names[7], gravityDirDefaults),								//gravityDir
	PRM_Template(PRM_FLT, 1, &names[8], PRMzeroDefaults),								//fluidStr

	PRM_Template(PRM_FLT, 1, &names[14], PRMzeroDefaults),				//noiseAmp
	PRM_Template(PRM_FLT, 1, &names[15], PRMoneDefaults, 0, &minZeroRange),				//noiseFreq
	PRM_Template(PRM_INT, 1, &names[16], PRMtwoDefaults, 0, &minOneRange),				//noiseOct
	PRM_Template(PRM_FLT, 1, &names[17], PRMtwoDefaults, 0, &minZeroRange),				//noiseLac
	PRM_Template(PRM_XYZ, 3, &names[18], PRMzeroDefaults),				//noiseOffset

	PRM_Template(PRM_TOGGLE, 1, &names[9], PRMoneDefaults),								//preview
	PRM_Template(PRM_FLT, 1, &names[10], PRMoneDefaults, 0, &pointSizeRange),				//pointsize
	PRM_Template(PRM_FLT, 1, &names[11], &opacityDefault, 0, &minZeroRange),				//opacity
	PRM_Template(PRM_RGB, 3, &names[12], startColorDefaults),							//startcol
	PRM_Template(PRM_RGB, 3, &names[13], endColorDefaults),								//endcol


    PRM_Template(),
};


OP_Node * SOP_CudaParticles::myConstructor(OP_Network *net, const char *name, OP_Operator *op) {
    return new SOP_CudaParticles(net, name, op);
}

SOP_CudaParticles::SOP_CudaParticles(OP_Network *net, const char *name, OP_Operator *op) : SOP_Node(net, name, op) {

	particlesSystem = new VHParticlesSystemHoudini();

	f = -1;
	oldf = -1;
	hSystemInit = 0;

}

SOP_CudaParticles::~SOP_CudaParticles() {

	size_t free, total;

	//cu::cudaThreadExit();

	cu::cudaMemGetInfo(&free, &total);
        
    //printf("mem = %lu %lu\n", free, total);

	delete particlesSystem;


}


OP_ERROR SOP_CudaParticles::cookMySop(OP_Context &context) {

	oldf = f;
	f =	context.getFrame();
	GEO_ParticleVertex* pvtx;

	double t = context.getTime();

	particlesSystem->dt = 1/(OPgetDirector()->getChannelManager()->getSamplesPerSec() * SUBSTEPS(t));
	particlesSystem->preview = PREVIEW(t);

	particlesSystem->partsLife = LIFE(t);
	particlesSystem->partsLifeVar = LIFEVAR(t);


	particlesSystem->velDamp = VELDAMP(t);
	particlesSystem->gravityStrength = GRAVITYSTR(t);
	particlesSystem->gravityDir = cu::make_float3(GRAVITYX(t),GRAVITYY(t),GRAVITYZ(t));
	particlesSystem->fluidStrength = FLUIDSTR(t);

	particlesSystem->noiseAmp = cu::make_float3(NOISEAMP(t),NOISEAMP(t),NOISEAMP(t));
	particlesSystem->noiseOct = NOISEOCT(t);
	particlesSystem->noiseFreq = NOISEFREQ(t);
	particlesSystem->noiseLac = NOISELACUN(t);
	particlesSystem->noiseOffset = cu::make_float3(NOISEOFFSETX(t),NOISEOFFSETY(t),NOISEOFFSETZ(t));

	particlesSystem->pointSize = POINTSIZE(t);
	particlesSystem->opacity = OPACITY(t);
	particlesSystem->startColor = cu::make_float3(STARTCOLORX(t),STARTCOLORY(t),STARTCOLORZ(t));
	particlesSystem->endColor = cu::make_float3(ENDCOLORX(t),ENDCOLORY(t),ENDCOLORZ(t));


	UT_Interrupt	*boss;

	OP_Node::flags().timeDep = 1;

    if (error() < UT_ERROR_ABORT) {
		boss = UTgetInterrupt();	

		// Start the interrupt server
		if (boss->opStart("Building Particles")){

			//gdp->clearAndDestroy();

			static float		 zero = 0.0;
			GB_AttributeRef partsAtt = gdp->addAttrib("cudaParticlesPreview", sizeof(int), GB_ATTRIB_INT, &zero);
			gdp->attribs().getElement().setValue<int>(partsAtt, particlesSystem->preview);

			GB_AttributeRef systemIdAtt = gdp->addAttrib("systemId", sizeof(int), GB_ATTRIB_INT, &zero);
			gdp->attribs().getElement().setValue<int>(systemIdAtt, particlesSystem->id);

			if (f < STARTFRAME(t)) {

				gdp->clearAndDestroy();
				particlesSystem->resetParticles();

			} else if (f == STARTFRAME(t)) {

				gdp->clearAndDestroy();
				particlesSystem->resetParticles();

				int maxParts = MAXPARTS(t);
				if (particlesSystem->nParts!=maxParts)
					particlesSystem->changeMaxParts(maxParts);

				//hSystem = (GEO_PrimParticle *)gdp->appendPrimitive(GEOPRIMPART);
				//hSystem->clearAndDestroy();

				GB_AttributeRef hVelocity = gdp->addPointAttrib("v", sizeof(UT_Vector3),GB_ATTRIB_VECTOR, 0);
				GB_AttributeRef hLife = gdp->addPointAttrib("life", sizeof(float)*2,GB_ATTRIB_FLOAT, 0);

				if(particlesSystem->preview!=1) {

					UT_Vector4 orig = UT_Vector4(0,0,0,1);

					

					for (int i = 0; i<particlesSystem->nParts; i++) {

						GEO_Point* newPoint = gdp->appendPoint();
						newPoint->setPos(orig);

						/*pvtx = hSystem->giveBirth();
						GEO_Point* ppt = pvtx->getPt();
						//ppt->getPos().assign(0,0,0,1);*/
						hSystemInit = 1;

					}
				}

			} else {

				if(particlesSystem->nParts != -1) {

					if(lockInputs(context) >= UT_ERROR_ABORT)
						return error();

					if(getInput(0)){

						GU_Detail* emittersInput = (GU_Detail*)inputGeo(0, context);
						GEO_PointList emittersList = emittersInput->points();
						int numEmitters = emittersList.entries();

						if (numEmitters != particlesSystem->nEmit) {
							delete particlesSystem->emitters;
							particlesSystem->nEmit = numEmitters;
							particlesSystem->emitters = new ParticlesEmitter[numEmitters];
						}

						GEO_AttributeHandle radAh, amountAh;
						GEO_AttributeHandle initVelAh, radVelAmpAh, noiseVelAmpAh,
							noiseVelOffsetAh, noiseVelOctAh, noiseVelLacAh, noiseVelFreqAh;

						radAh = emittersInput->getPointAttribute("radius");
						amountAh = emittersInput->getPointAttribute("amount");
						initVelAh = emittersInput->getPointAttribute("initVel");
						radVelAmpAh = emittersInput->getPointAttribute("radVelAmp");
						noiseVelAmpAh = emittersInput->getPointAttribute("noiseVelAmp");
						noiseVelOffsetAh = emittersInput->getPointAttribute("noiseVelOffset");
						noiseVelOctAh = emittersInput->getPointAttribute("noiseVelOct");
						noiseVelLacAh = emittersInput->getPointAttribute("noiseVelLac");
						noiseVelFreqAh = emittersInput->getPointAttribute("noiseVelFreq");

						for (int i = 0; i < numEmitters; i++) {

							UT_Vector4 emitPos = emittersList[i]->getPos();
							UT_Vector3 emitPos3(emitPos);

							particlesSystem->emitters[i].posX = emitPos.x();
							particlesSystem->emitters[i].posY = emitPos.y();
							particlesSystem->emitters[i].posZ = emitPos.z();

							radAh.setElement(emittersList[i]);
							amountAh.setElement(emittersList[i]);
							initVelAh.setElement(emittersList[i]);
							radVelAmpAh.setElement(emittersList[i]);
							noiseVelAmpAh.setElement(emittersList[i]);
							noiseVelOffsetAh.setElement(emittersList[i]);
							noiseVelOctAh.setElement(emittersList[i]);
							noiseVelLacAh.setElement(emittersList[i]);
							noiseVelFreqAh.setElement(emittersList[i]);

							particlesSystem->emitters[i].radius = radAh.getF(0);
							particlesSystem->emitters[i].amount = amountAh.getF(0);

							particlesSystem->emitters[i].velX = initVelAh.getF(0);
							particlesSystem->emitters[i].velY = initVelAh.getF(1);
							particlesSystem->emitters[i].velZ = initVelAh.getF(2);

							particlesSystem->emitters[i].radVelAmp = radVelAmpAh.getF(0);

							particlesSystem->emitters[i].noiseVelAmpX = noiseVelAmpAh.getF(0);
							particlesSystem->emitters[i].noiseVelAmpY = noiseVelAmpAh.getF(1);
							particlesSystem->emitters[i].noiseVelAmpZ = noiseVelAmpAh.getF(2);

							particlesSystem->emitters[i].noiseVelOffsetX = noiseVelOffsetAh.getF(0);
							particlesSystem->emitters[i].noiseVelOffsetY = noiseVelOffsetAh.getF(1);
							particlesSystem->emitters[i].noiseVelOffsetZ = noiseVelOffsetAh.getF(2);

							particlesSystem->emitters[i].noiseVelOct = noiseVelOctAh.getF(0);
							particlesSystem->emitters[i].noiseVelLac = noiseVelLacAh.getF(0);
							particlesSystem->emitters[i].noiseVelFreq = noiseVelFreqAh.getF(0);

						}
					} else {

						particlesSystem->nEmit = 0;

					}

					if(getInput(1)){

						GU_Detail* fluidInput = (GU_Detail*)inputGeo(1, context);

						GEO_AttributeHandle fluidIdAh= fluidInput->getDetailAttribute("solverId");
						fluidIdAh.setElement(fluidInput);

						int sId = fluidIdAh.getI();

						VHFluidSolver3D* curr3DSolver = VHFluidSolver3D::solverList[sId];

						particlesSystem->fluidSolver = curr3DSolver;

					}

						

					unlockInputs();


					if (f!=oldf) {

						particlesSystem->emitParticles();
						particlesSystem->updateParticles();

					}


					if(particlesSystem->preview!=1 && hSystemInit == 1) {

						cu::cudaMemcpy( particlesSystem->host_pos, particlesSystem->dev_pos,
							particlesSystem->nParts*sizeof(cu::float3), cu::cudaMemcpyDeviceToHost );

						GEO_Point* ppt;
						int i = 0;
						 UT_Vector4		p;

						FOR_ALL_GPOINTS(gdp, ppt) {

							ppt->getPos() = UT_Vector4(particlesSystem->host_pos[i*3],
													particlesSystem->host_pos[i*3+1],
													particlesSystem->host_pos[i*3+2],
													1);
							i++;

						}

						/*pvtx = hSystem->iterateInit();

						for (int i =0; i<particlesSystem->nParts; i++){
							pvtx->getPos().assign(particlesSystem->host_pos[i*3],
													particlesSystem->host_pos[i*3+1],
													particlesSystem->host_pos[i*3+2],
													1);
							pvtx = hSystem->iterateFastNext(pvtx);

						}*/

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