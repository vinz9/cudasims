#ifndef __SOP_FluidSolver2D_h__
#define __SOP_FluidSolver2D_h__

#include <SOP/SOP_Node.h>

#include <UT/UT_Interrupt.h>
#include <UT/UT_VoxelArray.h>

#include <PRM/PRM_Include.h>

#include <GU/GU_PrimVolume.h>
#include <GEO/GEO_Point.h>

#include <OP/OP_Director.h>
#include <OP/OP_Channels.h>

#include "vhFluidSolver2DHoudini.h"

namespace cu{
	#include <cuda_runtime_api.h>
	#include <cuda_gl_interop.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>
}

class SOP_FluidSolver2D : public SOP_Node {

public:
    static OP_Node		*myConstructor(OP_Network*, const char *,
							    OP_Operator *);

    static PRM_Template		 myTemplateList[];

	VHFluidSolver2DHoudini*   fluidSolver;


protected:
	     SOP_FluidSolver2D(OP_Network *net, const char *name, OP_Operator *op);
    virtual ~SOP_FluidSolver2D();

    virtual OP_ERROR		 cookMySop(OP_Context &context);

	float f, oldf;


private:

	float	POSX(float t) 	{ return evalFloat("pos", 0, t); }
	float	POSY(float t) 	{ return evalFloat("pos", 1, t); }
	float	POSZ(float t) 	{ return evalFloat("pos", 2, t); }


	float	ROTX(float t) 	{ return evalFloat("rot", 0, t); }
	float	ROTY(float t) 	{ return evalFloat("rot", 1, t); }
	float	ROTZ(float t) 	{ return evalFloat("rot", 2, t); }

	
	float	FLUIDSIZEX(float t) 	{ return evalFloat("size", 0, t); }
    float	FLUIDSIZEY(float t) 	{ return evalFloat("size", 1, t); }
    float	FLUIDSIZEZ(float t) 	{ return evalFloat("size", 2, t); }

	float	RESX(float t) 	{ return evalFloat("res", 0, t); }
    float	RESY(float t) 	{ return evalFloat("res", 1, t); }
    float	RESZ(float t) 	{ return evalFloat("res", 2, t); }

	int BORDERNEGX(float t) { return evalInt("borderNegX", 0, t); }
	int BORDERPOSX(float t) { return evalInt("borderPosX", 0, t); }
	int BORDERNEGY(float t) { return evalInt("borderNegY", 0, t); }
	int BORDERPOSY(float t) { return evalInt("borderPosY", 0, t); }

	int PREVIEW(float t) { return evalInt("preview", 0, t); }
	int PREVIEWTYPE(float t) { return evalInt("previewType", 0, t); }
	float BOUNDS(float t) { return evalFloat("bounds", 0, t); }


	int	SUBSTEPS(float t)	{ return evalInt("substeps", 0, t); }
	int	JACITER(float t)	{ return evalInt("jacIter", 0, t); }
	int	STARTFRAME(float t)	{ return evalInt("startFrame", 0, t); }

	float	DENSDIS(float t)	{ return evalFloat("densDis", 0, t); }
    float	DENSBUOYSTRENGTH(float t)	{ return evalFloat("densBuoyStrength", 0, t); }
	float	DENSBUOYDIRX(float t)	{ return evalFloat("densBuoyDir", 0, t); }
	float	DENSBUOYDIRY(float t)	{ return evalFloat("densBuoyDir", 1, t); }
	float	DENSBUOYDIRZ(float t)	{ return evalFloat("densBuoyDir", 2, t); }

	float	VELDAMP(float t)	{ return evalFloat("velDamp", 0, t); }
	float	VORTCONF(float t)	{ return evalFloat("vortConf", 0, t); }

	float	NOISESTR(float t)	{ return evalFloat("noiseStr", 0, t); }
	float	NOISEFREQ(float t)	{ return evalFloat("noiseFreq", 0, t); }
	int		NOISEOCT(float t)	{ return evalInt("noiseOct", 0, t); }
	float	NOISELACUN(float t)	{ return evalFloat("noiseLacun", 0, t); }
	float	NOISESPEED(float t)	{ return evalFloat("noiseSpeed", 0, t); }
	float	NOISEAMP(float t)	{ return evalFloat("noiseAmp", 0, t); }

};


#endif