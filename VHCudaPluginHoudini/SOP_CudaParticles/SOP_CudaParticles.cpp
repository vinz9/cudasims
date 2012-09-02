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
static PRM_Default	slicesDefault(64);
static PRM_Default	spritePathDefault(0, "C:/pictures/centerGradient.tif");

static PRM_Default	resDefaults[] = { PRM_Default(60),	PRM_Default(60) , PRM_Default(60)};


static PRM_Range    minOneRange(PRM_RANGE_RESTRICTED,1,PRM_RANGE_FREE,10);
static PRM_Range    minZeroRange(PRM_RANGE_RESTRICTED,0,PRM_RANGE_FREE,10);
static PRM_Range    zeroToOneRange(PRM_RANGE_RESTRICTED,0,PRM_RANGE_RESTRICTED,1);
static PRM_Range    pointSizeRange(PRM_RANGE_RESTRICTED,0,PRM_RANGE_FREE,10);
static PRM_Range    resMulRange(PRM_RANGE_RESTRICTED,0.1,PRM_RANGE_RESTRICTED,1);

static PRM_Name     switcherName("particlesSystemSwitcher");
static PRM_Default  switcherList[] = {
    PRM_Default(10, "Leads"),
	PRM_Default(13, "Trails"),
	PRM_Default(12, "Display"),
};

static PRM_Name         displayMode[] =
{
    PRM_Name("points", "Points"),
    PRM_Name("lines", "Lines"),
    PRM_Name("sprites", "Sprites"),
	PRM_Name("shadowedSprites", "Shadowed Sprites"),

    PRM_Name(0)
};
static PRM_ChoiceList   displayModeMenu(PRM_CHOICELIST_SINGLE, displayMode);

static PRM_Name         blendingMode[] =
{
    PRM_Name("add", "Additive"),
    PRM_Name("alpha", "Alpha"),

    PRM_Name(0)
};
static PRM_ChoiceList   blendingModeMenu(PRM_CHOICELIST_SINGLE, blendingMode);

// The names here have to match the inline evaluation functions
static PRM_Name        names[] = {

	PRM_Name("maxLeadParts",	"Max Lead Particles"),
	PRM_Name("startFrame",	"Start Frame"),
	PRM_Name("substeps",	"Substeps"),
	PRM_Name("life",	"Life"),
	PRM_Name("lifeVar",	"Life Var"),

	PRM_Name("velDamp",		"Velocity Damping"),
	PRM_Name("gravityStr",		"Gravity Strength"),
	PRM_Name("gravityDir",		"Gravity Dir"),
	PRM_Name("fluidStr",		"Fluid Strength"),

	PRM_Name("noiseScale","Noise Scale"),
	PRM_Name("noiseAmp","Noise Amp"),
	PRM_Name("noiseFreq","Noise Freq"),
	PRM_Name("noiseOct","Noise Oct"),
	PRM_Name("noiseLacun","Noise Lac"),
	PRM_Name("noiseOffset","Noise Offset"),


	PRM_Name("trailLength",		"Trail Length"),
	PRM_Name("inheritVel",		"Inherit Vel"),
	PRM_Name("inheritAge",		"Inherit Age"),

	PRM_Name("velDampTrail",		"Velocity Damping"),
	PRM_Name("gravityStrTrail",		"Gravity Strength"),
	PRM_Name("gravityDirTrail",		"Gravity Dir"),
	PRM_Name("fluidStrTrail",		"Fluid Strength"),

	PRM_Name("noiseScaleTrail","Noise Scale"),
	PRM_Name("noiseAmpTrail","Noise Amp"),
	PRM_Name("noiseFreqTrail","Noise Freq"),
	PRM_Name("noiseOctTrail","Noise Oct"),
	PRM_Name("noiseLacunTrail","Noise Lac"),
	PRM_Name("noiseOffsetTrail","Noise Offset"),


	PRM_Name("preview",	"Preview"),

	PRM_Name("displayMode",	"Display Mode"),
	PRM_Name("blendingMode", "Blending Mode"),
	PRM_Name("sortParts",	"Sort Particles"),
	PRM_Name("nSlices",	"Slices Number"),
	PRM_Name("resMul",	"Res Multiplier"),
	PRM_Name("spritePath",	"Sprite Path"),

	PRM_Name("pointSize","Point Size"),
	PRM_Name("lineWidth","Line Width"),
	PRM_Name("opacity","Opacity"),
	PRM_Name("startColor","Start Color"),
	PRM_Name("endColor","End Color"),




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

	PRM_Template(PRM_FLT, 1, &names[9], PRMzeroDefaults),				//noiseScale
	PRM_Template(PRM_XYZ, 3, &names[10], PRMoneDefaults),				//noiseAmp
	PRM_Template(PRM_FLT, 1, &names[11], PRMoneDefaults, 0, &minZeroRange),				//noiseFreq
	PRM_Template(PRM_INT, 1, &names[12], PRMtwoDefaults, 0, &minOneRange),				//noiseOct
	PRM_Template(PRM_FLT, 1, &names[13], PRMtwoDefaults, 0, &minZeroRange),				//noiseLac
	PRM_Template(PRM_XYZ, 3, &names[14], PRMzeroDefaults),				//noiseOffset

	PRM_Template(PRM_INT, 1, &names[15], PRMzeroDefaults, 0, &minZeroRange),				//trailLength

	PRM_Template(PRM_FLT, 1, &names[16], PRMzeroDefaults),				//inheritVel
	PRM_Template(PRM_FLT, 1, &names[17], PRMzeroDefaults),				//inheritAge

	PRM_Template(PRM_FLT, 1, &names[18], &velDampDefault, 0, &minZeroRange),				//velDamp
	PRM_Template(PRM_FLT, 1, &names[19], &gravityStrDefault),								//gravityStr
	PRM_Template(PRM_XYZ, 3, &names[20], gravityDirDefaults),								//gravityDir
	PRM_Template(PRM_FLT, 1, &names[21], PRMzeroDefaults),								//fluidStr

	PRM_Template(PRM_FLT, 1, &names[22], PRMzeroDefaults),				//noiseScale
	PRM_Template(PRM_XYZ, 3, &names[23], PRMoneDefaults),				//noiseAmp
	PRM_Template(PRM_FLT, 1, &names[24], PRMoneDefaults, 0, &minZeroRange),				//noiseFreq
	PRM_Template(PRM_INT, 1, &names[25], PRMtwoDefaults, 0, &minOneRange),				//noiseOct
	PRM_Template(PRM_FLT, 1, &names[26], PRMtwoDefaults, 0, &minZeroRange),				//noiseLac
	PRM_Template(PRM_XYZ, 3, &names[27], PRMzeroDefaults),				//noiseOffset

	PRM_Template(PRM_TOGGLE, 1, &names[28], PRMoneDefaults),								//preview
	PRM_Template(PRM_ORD, 1, &names[29], 0, &displayModeMenu),								//displayMode
	PRM_Template(PRM_ORD, 1, &names[30], 0, &blendingModeMenu),								//blendingMode
	PRM_Template(PRM_TOGGLE, 1, &names[31], PRMzeroDefaults),								//sortParts
	PRM_Template(PRM_INT, 1, &names[32], &slicesDefault,0, &minOneRange),					//nSlices
	PRM_Template(PRM_FLT, 1, &names[33], PRMoneDefaults, 0,&resMulRange),					//resMul
	PRM_Template(PRM_FILE, 1, &names[34], &spritePathDefault),								//spritePath
	PRM_Template(PRM_FLT, 1, &names[35], PRMoneDefaults, 0, &pointSizeRange),				//pointsize
	PRM_Template(PRM_FLT, 1, &names[36], PRMoneDefaults, 0, &pointSizeRange),				//lineWidth
	PRM_Template(PRM_FLT, 1, &names[37], &opacityDefault, 0, &zeroToOneRange),				//opacity
	PRM_Template(PRM_RGB, 3, &names[38], startColorDefaults),							//startcol
	PRM_Template(PRM_RGB, 3, &names[39], endColorDefaults),								//endcol


    PRM_Template(),
};

const char * SOP_CudaParticles::inputLabel(unsigned inum) const
{
    switch (inum)
    {
	case 0: return "Emitters";
	case 1:	return "Fluid";
	case 2: return "Attractors";
	case 3:	return "Lights";
    }
    return "Unknown source";
}

OP_Node * SOP_CudaParticles::myConstructor(OP_Network *net, const char *name, OP_Operator *op) {
    return new SOP_CudaParticles(net, name, op);
}

SOP_CudaParticles::SOP_CudaParticles(OP_Network *net, const char *name, OP_Operator *op) : SOP_Node(net, name, op) {

	particlesSystem = new VHParticlesSystemHoudini();

	//oldSpritePath.
	//oldSpritePath.harden();
	//oldSpritePath = "tada";

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

	particlesSystem->partsLife = LIFE(t);
	particlesSystem->partsLifeVar = LIFEVAR(t);


	((DampingForce*)(particlesSystem->leadsForces[0]))->strength = VELDAMP(t);

	((GravityForce*)(particlesSystem->leadsForces[1]))->strength = GRAVITYSTR(t);
	((GravityForce*)(particlesSystem->leadsForces[1]))->gravityDir = cu::make_float3(GRAVITYX(t),GRAVITYY(t),GRAVITYZ(t));

	((TurbulenceForce*)(particlesSystem->leadsForces[2]))->strength = NOISESCALE(t);
	((TurbulenceForce*)(particlesSystem->leadsForces[2]))->noiseAmp = cu::make_float3(NOISEAMPX(t),NOISEAMPY(t),NOISEAMPZ(t));
	((TurbulenceForce*)(particlesSystem->leadsForces[2]))->noiseOct = NOISEOCT(t);
	((TurbulenceForce*)(particlesSystem->leadsForces[2]))->noiseFreq = NOISEFREQ(t);
	((TurbulenceForce*)(particlesSystem->leadsForces[2]))->noiseLac = NOISELACUN(t);
	((TurbulenceForce*)(particlesSystem->leadsForces[2]))->noiseOffset = cu::make_float3(NOISEOFFSETX(t),NOISEOFFSETY(t),NOISEOFFSETZ(t));

	((FluidForce*)(particlesSystem->leadsForces[3]))->strength = FLUIDSTR(t);

	particlesSystem->inheritVel = INHERITVEL(t);
	particlesSystem->inheritAge = INHERITAGE(t);

	((DampingForce*)(particlesSystem->trailsForces[0]))->strength = VELDAMPTR(t);

	((GravityForce*)(particlesSystem->trailsForces[1]))->strength = GRAVITYSTRTR(t);
	((GravityForce*)(particlesSystem->trailsForces[1]))->gravityDir = cu::make_float3(GRAVITYTRX(t),GRAVITYTRY(t),GRAVITYTRZ(t));

	((TurbulenceForce*)(particlesSystem->trailsForces[2]))->strength = NOISESCALETR(t);
	((TurbulenceForce*)(particlesSystem->trailsForces[2]))->noiseAmp = cu::make_float3(NOISEAMPTRX(t),NOISEAMPTRY(t),NOISEAMPTRZ(t));
	((TurbulenceForce*)(particlesSystem->trailsForces[2]))->noiseOct = NOISEOCTTR(t);
	((TurbulenceForce*)(particlesSystem->trailsForces[2]))->noiseFreq = NOISEFREQTR(t);
	((TurbulenceForce*)(particlesSystem->trailsForces[2]))->noiseLac = NOISELACUNTR(t);
	((TurbulenceForce*)(particlesSystem->trailsForces[2]))->noiseOffset = cu::make_float3(NOISEOFFSETTRX(t),NOISEOFFSETTRY(t),NOISEOFFSETTRZ(t));

	((FluidForce*)(particlesSystem->trailsForces[3]))->strength = FLUIDSTRTR(t);

	particlesSystem->preview = PREVIEW(t);

	particlesSystem->pRend->displayMode = DISPLAYMODE(t);
	particlesSystem->pRend->blendingMode = BLENDINGMODE(t);
	particlesSystem->pRend->sortParts = SORTPARTS(t);
	particlesSystem->pRend->nSlices = NSLICES(t);
	particlesSystem->pRend->resMul = RESMUL(t);


	UT_String spritePath;
	SPRITEPATH(spritePath, t);

	UT_String oldSprite = particlesSystem->pRend->spritePath;

	if (spritePath != oldSprite) {
		particlesSystem->lockOpenGLContext();
		particlesSystem->pRend->loadSprite(spritePath);
		particlesSystem->unlockOpenGLContext();
		particlesSystem->pRend->spritePath = (char*)(spritePath.buffer());
	}

	particlesSystem->pRend->pointSize = POINTSIZE(t);
	particlesSystem->pRend->lineWidth = LINEWIDTH(t);

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

				int maxLeadParts = MAXLEADPARTS(t);
				int trailLength = TRAILLENGTH(t)+1;

				if (particlesSystem->nLeadParts!=maxLeadParts || particlesSystem->trailLength!= trailLength)
					particlesSystem->changeMaxParts(maxLeadParts,trailLength);

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

							particlesSystem->emitters[i].pos = cu::make_float3(emitPos.x(),emitPos.y(),emitPos.z());

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

							particlesSystem->emitters[i].vel = cu::make_float3(initVelAh.getF(0), initVelAh.getF(1), initVelAh.getF(2));

							particlesSystem->emitters[i].radVelAmp = radVelAmpAh.getF(0);

							particlesSystem->emitters[i].noiseVelAmp = cu::make_float3(noiseVelAmpAh.getF(0),noiseVelAmpAh.getF(1),noiseVelAmpAh.getF(2));

							particlesSystem->emitters[i].noiseVelOffset = cu::make_float3(noiseVelOffsetAh.getF(0),noiseVelOffsetAh.getF(1),noiseVelOffsetAh.getF(2));

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

						((FluidForce*)(particlesSystem->leadsForces[3]))->fluidSolver = curr3DSolver;
						((FluidForce*)(particlesSystem->trailsForces[3]))->fluidSolver = curr3DSolver;

					} else {

						((FluidForce*)(particlesSystem->leadsForces[3]))->fluidSolver = NULL;
						((FluidForce*)(particlesSystem->trailsForces[3]))->fluidSolver = NULL;


					}

					if(getInput(2)){

						GU_Detail* attractorsInput = (GU_Detail*)inputGeo(2, context);
						GEO_PointList attractorsList = attractorsInput->points();

						GEO_AttributeHandle strengthLeadsAh, strengthTrailsAh, radiusAh, decayAh;

						strengthLeadsAh = attractorsInput->getPointAttribute("strengthLeads");
						strengthTrailsAh = attractorsInput->getPointAttribute("strengthTrails");
						radiusAh = attractorsInput->getPointAttribute("radius");
						decayAh = attractorsInput->getPointAttribute("decay");

						int newNumAttractors = attractorsList.entries();
						int oldNumAttractors = particlesSystem->nLeadsForces-4;

						if (newNumAttractors != oldNumAttractors) {
							for (int i = 0; i<oldNumAttractors; i++) {
								delete particlesSystem->leadsForces[4+i];
								delete particlesSystem->trailsForces[4+i];
							}
							for (int i = 0; i<newNumAttractors; i++) {
								particlesSystem->leadsForces[4+i] = new AttractorForce(particlesSystem);
								particlesSystem->trailsForces[4+i] = new AttractorForce(particlesSystem);
							}
							particlesSystem->nLeadsForces = 4+newNumAttractors;
							particlesSystem->nTrailsForces = 4+newNumAttractors;
						}
						
						for (int i = 0; i<newNumAttractors; i++) {

							UT_Vector4 attOrigin = attractorsList[i]->getPos();
							cu::float3 attPos = cu::make_float3(attOrigin.x(),attOrigin.y(),attOrigin.z());
							((AttractorForce*)(particlesSystem->leadsForces[4+i]))->origin = attPos;
							((AttractorForce*)(particlesSystem->trailsForces[4+i]))->origin = attPos;

							strengthLeadsAh.setElement(attractorsList[i]);
							strengthTrailsAh.setElement(attractorsList[i]);
							radiusAh.setElement(attractorsList[i]);
							decayAh.setElement(attractorsList[i]);

							((AttractorForce*)(particlesSystem->leadsForces[4+i]))->strength = strengthLeadsAh.getF();
							((AttractorForce*)(particlesSystem->trailsForces[4+i]))->strength = strengthTrailsAh.getF();

							((AttractorForce*)(particlesSystem->leadsForces[4+i]))->radius = radiusAh.getF();
							((AttractorForce*)(particlesSystem->trailsForces[4+i]))->radius = radiusAh.getF();
							((AttractorForce*)(particlesSystem->leadsForces[4+i]))->decay = decayAh.getI();
							((AttractorForce*)(particlesSystem->trailsForces[4+i]))->decay = decayAh.getI();
						}


					} else {

							particlesSystem->nLeadsForces = 4;
							particlesSystem->nTrailsForces = 4;


					}

					if(getInput(3)){

						GU_Detail* lightsInput = (GU_Detail*)inputGeo(3, context);
						GEO_PointList lightsList = lightsInput->points();

						GEO_AttributeHandle lightColorAh, colorAttAh, shadowAlphaAh;
						GEO_AttributeHandle doBlurAh, blurRadiusAh, displayLightBufferAh, displayVectorsAh;

						lightColorAh = lightsInput->getPointAttribute("lightColor");
						colorAttAh = lightsInput->getPointAttribute("colorAtt");
						shadowAlphaAh = lightsInput->getPointAttribute("shadowAlpha");
						doBlurAh = lightsInput->getPointAttribute("doBlur");
						blurRadiusAh = lightsInput->getPointAttribute("blurRadius");
						displayLightBufferAh = lightsInput->getPointAttribute("displayLightBuffer");
						displayVectorsAh = lightsInput->getPointAttribute("displayVectors");

						UT_Vector4 lightPos = lightsList[0]->getPos();
						particlesSystem->pRend->lightPos = vec3f(lightPos.x(),lightPos.y(),lightPos.z());

						UT_Vector4 lightTarget = lightsList[1]->getPos();
						particlesSystem->pRend->lightTarget = vec3f(lightTarget.x(),lightTarget.y(),lightTarget.z());

						lightColorAh.setElement(lightsList[0]);
						colorAttAh.setElement(lightsList[0]);
						shadowAlphaAh.setElement(lightsList[0]);
						doBlurAh.setElement(lightsList[0]);
						blurRadiusAh.setElement(lightsList[0]);
						displayLightBufferAh.setElement(lightsList[0]);
						displayVectorsAh.setElement(lightsList[0]);

						particlesSystem->pRend->lightColor = vec3f(lightColorAh.getF(0), lightColorAh.getF(1), lightColorAh.getF(2));
						particlesSystem->pRend->colorAttenuation = vec3f(colorAttAh.getF(0), colorAttAh.getF(1), colorAttAh.getF(2));
						particlesSystem->pRend->shadowAlpha = shadowAlphaAh.getF();
						particlesSystem->pRend->doBlur = doBlurAh.getI();
						particlesSystem->pRend->blurRadius = blurRadiusAh.getF();
						particlesSystem->pRend->displayLightBuffer = displayLightBufferAh.getI();
						particlesSystem->pRend->displayVectors = displayVectorsAh.getI();


					} else {


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