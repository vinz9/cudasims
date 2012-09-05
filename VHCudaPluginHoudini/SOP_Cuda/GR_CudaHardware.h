#ifndef __GR_CudaHardware_h__
#define __GR_CudaHardware_h__

#include <RE/RE_Render.h>

#include <GEO/GEO_Primitive.h>

#include <GU/GU_Detail.h>
#include <GU/GU_PrimGroupClosure.h>

#include <GR/GR_Detail.h>
#include <GR/GR_RenderHook.h>

#include <GR/GR_DisplayOption.h>

namespace cu{
	#include <cuda_runtime_api.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>
}

class GR_CudaHardware : public GR_RenderHook
{
public:
    GR_CudaHardware();
    virtual ~GR_CudaHardware();


	static int preview(const GU_Detail *gdp);

#ifdef HOUDINI_11

	int		 getWireMask(GU_Detail *gdp, const GR_DisplayOption * ) const {
		if (preview(gdp))
			return 0;
		else
			return GEOPRIMALL;
    }

    int		 getShadedMask(GU_Detail *gdp, const GR_DisplayOption * ) const {
		if (preview(gdp))
			return 0;
		else
			return GEOPRIMALL;
    }

#else
	virtual GA_PrimCompat::TypeMask
        getWireMask(GU_Detail * /*gdp*/, const GR_DisplayOption *dopt) const
    {
        // If hull display is on, we don't draw anything.
        if (dopt->drawPrimHulls())
            return GEO_PrimTypeCompat::GEOPRIMNONE;
        else
            return GEO_PrimTypeCompat::GEOPRIMALL;
    }

    virtual GA_PrimCompat::TypeMask
        getShadedMask(GU_Detail * /*gdp*/, const GR_DisplayOption *dopt) const
    {
        // If hull display is on, we don't draw anything.
        if (dopt->drawPrimHulls())
            return GEO_PrimTypeCompat::GEOPRIMNONE;
        else
            return GEO_PrimTypeCompat::GEOPRIMALL;
    }
#endif

	virtual void renderWire(GU_Detail *gdp,
			RE_Render &ren,
			const GR_AttribOffset &ptinfo,
			const GR_DisplayOption *dopt,
			float lod,
			const GU_PrimGroupClosure *hidden_geometry);

    virtual void renderShaded(GU_Detail *gdp,
			RE_Render &ren,
			const GR_AttribOffset &ptinfo,
			const GR_DisplayOption *dopt,
			float lod,
			const GU_PrimGroupClosure *hidden_geometry);

    virtual const char *getName() const { return "GR_CudaHardware"; }

	virtual GR_HookRender getHookRenderType(GU_Detail *, const GR_DisplayOption * /*dopt*/){
		return VIEW_DEPENDENT;
	}

	virtual GR_HookOpacity getTransparencyMode(GU_Detail * ,
                                               const GR_DisplayOption* )
                           { return OPACITY_TRANSPARENT_NON_DEPTH_PEEL; }

	void drawWireCube(float x, float y, float z, RE_Render &ren);

};

#endif