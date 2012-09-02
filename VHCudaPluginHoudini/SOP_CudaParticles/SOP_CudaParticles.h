#ifndef __SOP_CudaParticles_h__
#define __SOP_CudaParticles_h__

#include <SOP/SOP_Node.h>

#include <UT/UT_Interrupt.h>
#include <UT/UT_VoxelArray.h>

#include <PRM/PRM_Include.h>

#include <GU/GU_PrimVolume.h>
#include <GEO/GEO_Point.h>
#include <GEO/GEO_PrimPart.h>

#include <OP/OP_Director.h>
#include <OP/OP_Channels.h>

#include "vhParticlesSystemHoudini.h"

namespace cu{
	#include <cuda_runtime_api.h>
	#include <vector_types.h>
	#include <cuda_gl_interop.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>
}

class SOP_CudaParticles : public SOP_Node {

public:
    static OP_Node		*myConstructor(OP_Network*, const char *,
							    OP_Operator *);

    static PRM_Template		 myTemplateList[];

protected:
	     SOP_CudaParticles(OP_Network *net, const char *name, OP_Operator *op);
    virtual ~SOP_CudaParticles();


    virtual OP_ERROR		 cookMySop(OP_Context &context);

	GEO_PrimParticle* hSystem;
	int hSystemInit;

	VHParticlesSystemHoudini* particlesSystem;

	int oldf;
	int f;


private:

	int	MAXPARTS(float t)	{ return evalInt("maxParts", 0, t); }

	int	STARTFRAME(float t)	{ return evalInt("startFrame", 0, t); }
	int	SUBSTEPS(float t)	{ return evalInt("substeps", 0, t); }

	float	LIFE(float t)	{ return evalFloat("life", 0, t); }
	float	LIFEVAR(float t)	{ return evalFloat("lifeVar", 0, t); }

	float	VELDAMP(float t)	{ return evalFloat("velDamp", 0, t); }
	float	GRAVITYSTR(float t)	{ return evalFloat("gravityStr", 0, t); }
	float	GRAVITYX(float t)	{ return evalFloat("gravityDir", 0, t); }
	float	GRAVITYY(float t)	{ return evalFloat("gravityDir", 1, t); }
	float	GRAVITYZ(float t)	{ return evalFloat("gravityDir", 2, t); }
	float	FLUIDSTR(float t)	{ return evalFloat("fluidStr", 0, t); }

	int	PREVIEW(float t)	{ return evalInt("preview", 0, t); }
	float	POINTSIZE(float t)	{ return evalFloat("pointSize", 0, t); }
	float	OPACITY(float t)	{ return evalFloat("opacity", 0, t); }

	float	STARTCOLORX(float t)	{ return evalFloat("startColor", 0, t); }
	float	STARTCOLORY(float t)	{ return evalFloat("startColor", 1, t); }
	float	STARTCOLORZ(float t)	{ return evalFloat("startColor", 2, t); }

	float	ENDCOLORX(float t)	{ return evalFloat("endColor", 0, t); }
	float	ENDCOLORY(float t)	{ return evalFloat("endColor", 1, t); }
	float	ENDCOLORZ(float t)	{ return evalFloat("endColor", 2, t); }


	float	NOISEAMP(float t)	{ return evalFloat("noiseAmp", 0, t); }
	float	NOISEFREQ(float t)	{ return evalFloat("noiseFreq", 0, t); }
	int		NOISEOCT(float t)	{ return evalInt("noiseOct", 0, t); }
	float	NOISELACUN(float t)	{ return evalFloat("noiseLacun", 0, t); }
	float	NOISEOFFSETX(float t)	{ return evalFloat("noiseOffset", 0, t); }
	float	NOISEOFFSETY(float t)	{ return evalFloat("noiseOffset", 1, t); }
	float	NOISEOFFSETZ(float t)	{ return evalFloat("noiseOffset", 2, t); }

	int	DRAWCUBE(float t)	{ return evalInt("drawCube", 0, t); }


};


#endif