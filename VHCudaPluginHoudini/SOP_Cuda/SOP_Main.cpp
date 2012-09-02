#pragma warning(disable : 4018)
#pragma warning(disable : 4244)
#pragma warning(disable : 4996)
#pragma warning(disable : 4800)
#pragma warning(disable : 4312)

#include <UT/UT_DSOVersion.h>
#include <UT/UT_Interrupt.h>

#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>

#include <RE/RE_Render.h>
#include <GR/GR_RenderHook.h>
#include <GR/GR_RenderTable.h>

#include "../SOP_FluidSolver/SOP_FluidSolver2D.h"
#include "../SOP_FluidSolver/SOP_FluidSolver3D.h"
#include "../SOP_CudaParticles/SOP_CudaParticles.h"
#include "GR_CudaHardware.h"

void newSopOperator(OP_OperatorTable *table) {
    table->addOperator(
	    new OP_Operator("vhFluidSolver2D",			// Internal name
			    "CudaFluidSolver2D",			// UI name
			     SOP_FluidSolver2D::myConstructor,	// How to build the SOP
			     SOP_FluidSolver2D::myTemplateList,	// My parameters
			     1,				// Min # of sources
			     2,				// Max # of sources
			     0,
				 OP_FLAG_GENERATOR)		// Flag it as generator
	    );


	table->addOperator(
	    new OP_Operator("vhFluidSolver3D",			// Internal name
			    "CudaFluidSolver3D",			// UI name
			     SOP_FluidSolver3D::myConstructor,	// How to build the SOP
			     SOP_FluidSolver3D::myTemplateList,	// My parameters
			     1,				// Min # of sources
			     2,				// Max # of sources
			     0,
				 OP_FLAG_GENERATOR)		// Flag it as generator
	    );

	table->addOperator(
	    new OP_Operator("vhCudaParticles",			// Internal name
			    "CudaParticles",			// UI name
			     SOP_CudaParticles::myConstructor,	// How to build the SOP
			     SOP_CudaParticles::myTemplateList,	// My parameters
			     1,				// Min # of sources
			     2,				// Max # of sources
				 0,
				 OP_FLAG_GENERATOR)

	    );
}

void newRenderHook(GR_RenderTable *table)
{

	GR_CudaHardware *cudaHook = new GR_CudaHardware;
    table->addHook(cudaHook, GR_RENDER_HOOK_VERSION);

}
