#include <stdio.h>
#include <cutil_math.h>

#include "perlinKernel3d.cu";

#define PI 3.1415926535897932f

texture<float,3>  texDens;
texture<float,3>  texNoise;
texture<float4,3>  texVel;
texture<float,3>  texDiv;
texture<float,3>  texPressure;
texture<float4,3>  texObstacles;
texture<float4,3>  texVort;


typedef struct {
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__global__ void addFluidForceKernel(float3* vel, float3* pos, cudaExtent gres, float3 invSize, float strength,
									float dt, int nParts)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int n = x;

	if (n<nParts) {

		float4 sample = tex3D(texVel, (pos[n].x*invSize.x+0.5f) * (gres.width)+0.5,
										(pos[n].y*invSize.y+0.5f) * (gres.height)+0.5,
										(pos[n].z*invSize.z+0.5f) * (gres.depth)+0.5);
		vel[n] = vel[n] + strength * make_float3(sample.x,sample.y,sample.z);

	}
}

__global__ void d_render(float4 *d_output, uint imageW, uint imageH, float density, cudaExtent gres,
						 float focalLength, float3 boxMin, float3 boxMax, float3 invSize, float maxSize, float stepMul)
{
    const float opacityThreshold = 0.99f;

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, focalLength));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
    if (!hit) return;
	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

	//float dist = tfar - tnear;
	float tstep = 0.01f*maxSize*stepMul;

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;

	int maxSteps = 1000;

    for(int i=0; i<maxSteps; i++) {
        // read from 3D texture
        // remap position to coordinates

		float sample = tex3D(texDens, (pos.x*invSize.x+0.5f) * (gres.width)+0.5,
										(pos.y*invSize.y+0.5f) * (gres.height)+0.5,
										(pos.z*invSize.z+0.5f) * (gres.depth)+0.5);

		//sample = (1-pow((float)(1-sample*density),(float)(tstep/0.1)));
		//sample = clamp((float)sample,0.0f,1.0f);
		sample = density * sample;

		float4 col = make_float4(sample,sample,sample,sample);


        // "under" operator for back-to-front blending
        //sum = lerp(sum, col, col.w);

        // pre-multiply alpha
       /* col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;*/
        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);
	

        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;
        if (t > tfar) break;

        pos += step;
    }
 
	d_output[y*imageW + x] = sum;
	//d_output[y*imageW + x] = make_float4(0.5,0,0,1);
	
}

__global__ void d_render_shadows(float4 *d_output, uint imageW, uint imageH, float density, cudaExtent gres,
						 float focalLength, float3 boxMin, float3 boxMax, float3 invSize, float3 size,
						 float maxSize, float3 lightPos, float stepMul, float shadowStepMul, float shadowThres, float shadowDens)
{
    const float opacityThreshold = 0.99f;

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, focalLength));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
    if (!hit) return;
	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

	//float dist = tfar - tnear;
	float tstep = 0.01f*maxSize*stepMul;

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;

	int maxSteps = 1000;

    for(int i=0; i<maxSteps; i++) {
        // read from 3D texture
        // remap position to coordinates

		float sample = tex3D(texDens, (pos.x*invSize.x+0.5f) * (gres.width)+0.5,
										(pos.y*invSize.y+0.5f) * (gres.height)+0.5,
										(pos.z*invSize.z+0.5f) * (gres.depth)+0.5);

		//sample = (1-pow((float)(1-sample*density),(float)(tstep/0.1)));
		//sample = clamp((float)sample,0.0f,1.0f);
		sample = density * sample;

		float4 col = make_float4(sample,sample,sample,sample);

		//float3 modelSpaceLight =  mul(c_invViewMatrix, lightPos);
		float3 modelSpaceLight =  lightPos;
		float3 lightRayStep = normalize(modelSpaceLight-pos)*tstep*shadowStepMul;

		float3 shadowPos = pos;
		float opa = 0;
		float opaAcc = 0;

		//float shadowThreshold = 0.9;

		for(int j = 0; j<maxSteps;j++) {
			opa = tex3D(texDens, (shadowPos.x*invSize.x+0.5f) * (gres.width)+0.5,
										(shadowPos.y*invSize.y+0.5f) * (gres.height)+0.5,
										(shadowPos.z*invSize.z+0.5f) * (gres.depth)+0.5);
			opaAcc = opaAcc + (1.0 - opaAcc) * opa;
			shadowPos = shadowPos + lightRayStep;

			if (opaAcc > shadowThres)
				break;

			if (shadowPos.x>size.x*0.5 || shadowPos.x<-size.x*0.5
				|| shadowPos.y>size.y*0.5 || shadowPos.y<-size.y*0.5
				|| shadowPos.z>size.z*0.5 || shadowPos.z<-size.z*0.5)
				break;

		}

		opaAcc = shadowDens*(clamp((float)(opaAcc) / (float)(shadowThres), 0.0f, 1.0f));

		col.x = (1-opaAcc)*col.x;
		col.y = (1-opaAcc)*col.y;
		col.z = (1-opaAcc)*col.z;



        // "under" operator for back-to-front blending
        //sum = lerp(sum, col, col.w);

        // pre-multiply alpha
       /* col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;*/
        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);
	

        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;
        if (t > tfar) break;

        pos += step;
    }
 
	d_output[y*imageW + x] = sum;
}

__global__ void createBorder(float4 *obst, cudaExtent gres, int posX, int negX, int posY, int negY, int posZ, int negZ, int bx) {
    // map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float4* pixel = obst + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {

		*pixel = make_float4(0,0,0,0);

		//if (x==0 || x==(gres.width-1) || y==0 || y==(gres.height-1) || z==0 || z==(gres.depth-1))

				if (negX == 1 && x == 0)
			pixel->w = 1;

				if (posX == 1 && x==(gres.width-1))
			pixel->w = 1;

				if (negY == 1 && y == 0)
			pixel->w = 1;

				if (posY == 1 && y==(gres.height-1))
			pixel->w = 1;

				if (negZ == 1 && z == 0)
			pixel->w = 1;

				if (posZ == 1 && z==(gres.depth-1))
			pixel->w = 1;

	}
}

__global__ void addCollider(float4 *obst, float radius, float3 position, cudaExtent gres, int bx, float3 vel) {
    // map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float4* pixel = obst + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {

		float3 coords = make_float3(x,y,z);
		float3 pos = (position - coords);
		float scaledRadius = radius;

		if (dot(pos,pos)<(scaledRadius*scaledRadius)) {
			pixel->x = vel.x;
			pixel->y = vel.y;
			pixel->z = vel.z;
			pixel->w = 1;


		}
	}

	
}

__global__ void advectVel(float4 *vel, float timestep, float dissipation, float3 invGridSize, cudaExtent gres, int bx) {
    // map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float4* pixel = vel + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {

		float xc = x+0.5;
		float yc = y+0.5;
		float zc = z+0.5;

		float solid = tex3D(texObstacles,xc,yc,zc).w;

		if (solid > 0) {
			*pixel = make_float4(0,0,0,0);
			return;
		}

		float3 coords = make_float3(xc,yc,zc);
		float4 oldVel = tex3D(texVel,xc,yc,zc);
		float3 pos = coords - timestep * invGridSize * make_float3(oldVel.x,oldVel.y,oldVel.z) * make_float3((float)gres.width,(float)gres.height,(float)gres.depth);

		float4 newVel = tex3D(texVel, pos.x,pos.y,pos.z);

		*pixel = (1-dissipation*timestep) * make_float4(newVel.x,newVel.y,newVel.z,0);

	}



}

__global__ void advectDens(float *dens, float timestep, float dissipation, float3 invGridSize, cudaExtent gres, int bx) {
    // map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float* pixel = dens + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {

		float xc = x+0.5;
		float yc = y+0.5;
		float zc = z+0.5;

		float solid = tex3D(texObstacles,xc,yc,zc).w;

		if (solid > 0) {
			*pixel = 0;
			return;
		}

		float3 coords = make_float3(xc,yc,zc);
		float4 oldVel = tex3D(texVel,xc,yc,zc);
		float3 pos = coords - timestep * invGridSize * make_float3(oldVel.x,oldVel.y,oldVel.z) * make_float3((float)gres.width,(float)gres.height,(float)gres.depth);

		*pixel = (1-dissipation*timestep) * tex3D(texDens, pos.x,pos.y,pos.z);
	}

	
}


__global__ void addDens(float* dens, float timestep, float radius, float amount, float3 position, cudaExtent gres, int bx) {
    // map from threadIdx/BlockIdx to pixel position

	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float* pixel = dens + z*gres.width*gres.height + y*gres.width + x;
	//*pixel = z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {

		float3 coords = make_float3(x,y,z);
		float3 pos = (position - coords);

		if (dot(pos,pos)<(radius*radius))
			*pixel += timestep*amount;
			//*pixel = 1.0;

		//*pixel = 1.0;

	
	}

}

__global__ void addDensBuoy(float4 *vel, float timestep, float strength, float3 dir, cudaExtent gres, int bx) {
    // map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float4* pixel = vel + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {

		float xc = x+0.5;
		float yc = y+0.5;
		float zc = z+0.5;

		float4 dir4 = make_float4(dir.x,dir.y,dir.z,0);

		float dens = tex3D(texDens,xc,yc,zc);

		*pixel += timestep * strength * dir4 * tex3D(texDens,xc,yc,zc);
	}


}

//Simple kernel fills an array with perlin noise
__global__ void k_perlin(float* noise, cudaExtent gres, float3 delta,
			 float time, int octaves, float lacun, float gain, float freq, float amp, int bx) {


	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	float xCur = (float)x * delta.x;
	float yCur = (float)y * delta.y;
	float zCur = (float)z * delta.z;

	// get a pointer to this pixel
	float* pixel = noise + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {
		*pixel = noise1D(xCur, yCur-time, zCur, octaves, lacun, gain, freq, amp);

	}
}

__global__ void addNoise(float4 *vel, float timestep, float strength, cudaExtent gres, int bx) {
    // map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float4* pixel = vel + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {
	
		float xc = x+0.5;
		float yc = y+0.5;
		float zc = z+0.5;

		float noise = strength*timestep*tex3D(texNoise,xc,yc,zc)*tex3D(texDens,xc,yc,zc);
		
		*pixel += make_float4(noise,noise,noise,0);

	}


	
}

__global__ void vorticity(float4 *vort, cudaExtent gres, float3 invCellSize, int bx) {
    // map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float4* pixel = vort + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {

		float xc = x+0.5;
		float yc = y+0.5;
		float zc = z+0.5;

		float4 vL = tex3D(texVel,xc-1,yc,zc);
		float4 vR = tex3D(texVel,xc+1,yc,zc);
		float4 vT = tex3D(texVel,xc,yc+1,zc);
		float4 vB = tex3D(texVel,xc,yc-1,zc);
		float4 vBa = tex3D(texVel,xc,yc,zc-1);
		float4 vF = tex3D(texVel,xc,yc,zc+1);

		//obstacles
		float4 oL = tex3D(texObstacles,xc-1,yc,zc);
		float4 oR = tex3D(texObstacles,xc+1,yc,zc);
		float4 oT = tex3D(texObstacles,xc,yc+1,zc);
		float4 oB = tex3D(texObstacles,xc,yc-1,zc);
		float4 oBa = tex3D(texObstacles,xc,yc,zc-1);
		float4 oF = tex3D(texObstacles,xc,yc,zc+1);

		// Use obstacle velocities for solid cells:
		if (oL.w>0) vL = make_float4(oL.x,oL.y,oL.z,0);
		if (oR.w>0) vR = make_float4(oR.x,oR.y,oR.z,0);
		if (oT.w>0) vT = make_float4(oT.x,oT.y,oT.z,0);
		if (oB.w>0) vB = make_float4(oB.x,oB.y,oT.z,0);
		if (oT.w>0) vBa = make_float4(oBa.x,oBa.y,oBa.z,0);
		if (oB.w>0) vF = make_float4(oF.x,oF.y,oF.z,0);

		*pixel = 0.5 * make_float4(invCellSize.y*(vT.z-vB.z)-invCellSize.z*(vF.y-vBa.y),
									invCellSize.z*(vF.x-vBa.x)-invCellSize.x*(vL.z-vR.z),
									invCellSize.x*(vR.y - vL.y) - invCellSize.y*(vT.x - vB.x),0);

	}

	
}

__global__ void vortConf(float4 *vel, float timestep, float strength, cudaExtent gres, int bx) {
    // map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float4* pixel = vel + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {

		float xc = x+0.5;
		float yc = y+0.5;
		float zc = z+0.5;

		float4 vortL = tex3D(texVort,xc-1,yc,zc);
		float vortLS = length(make_float3(vortL.x,vortL.y,vortL.z));

		float4 vortR = tex3D(texVort,xc+1,yc,zc);
		float vortRS = length(make_float3(vortR.x,vortR.y,vortR.z));

		float4 vortT = tex3D(texVort,xc,yc+1,zc);
		float vortTS = length(make_float3(vortT.x,vortT.y,vortT.z));

		float4 vortB = tex3D(texVort,xc,yc-1,zc);
		float vortBS = length(make_float3(vortB.x,vortB.y,vortB.z));

		float4 vortBa = tex3D(texVort,xc,yc,zc-1);
		float vortBaS = length(make_float3(vortBa.x,vortBa.y,vortBa.z));

		float4 vortF = tex3D(texVort,xc,yc,zc+1);
		float vortFS = length(make_float3(vortF.x,vortF.y,vortF.z));

		float4 vortC = tex3D(texVort,xc,yc,zc);

		const float EPSILON = 2.4414e-4; // 2^-12
		float3 eta = 0.5 * make_float3(gres.width*(vortRS-vortLS),
										gres.height*(vortTS-vortBS),
										gres.depth*(vortFS-vortBaS));

		eta = normalize(eta+make_float3(EPSILON,EPSILON,EPSILON));

		float3 force = make_float3(eta.y*vortC.z - eta.z*vortC.y,
									eta.z*vortC.x - eta.x*vortC.z,
									eta.x*vortC.y - eta.y*vortC.x);

		force = force * strength * timestep;

		*pixel += make_float4(force.x,force.y,force.z,0);

	}

	
}

__global__ void addVel(float4 *vel, float strength, cudaExtent gres, int bx) {
    // map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float4* pixel = vel + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {

		float3 coords = make_float3(x,y,z);
		float3 position = make_float3(32,32,32);
		float3 pos = (position - coords);

		float radius = 20;

		float xc = x+0.5;
		float yc = y+0.5;
		float zc = z+0.5;

		float4 oldVel = tex3D(texVel,xc,yc,zc);

		if (dot(pos,pos)<(radius*radius))
			//dens[offset] += timestep*amount;
			//*pixel += timestep*amount;

		*pixel = oldVel+0.001*make_float4(0,1,0,0);
	}


}

/*__global__ void addNoisyDens(float* dens, float radius, float amount, float3 position, cudaExtent gres) {
    // map from threadIdx/BlockIdx to pixel position

	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float* pixel = dens + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {

		float3 coords = make_float3(x,y,z);
		float3 pos = (position - coords);

		if (dot(pos,pos)<(radius*radius))
			*pixel = 1.0f * tex3D(texNoise, x+0.5, y+0.5, z+0.5);
	}

}*/

__global__ void divergence(float *div, cudaExtent gres, float3 invCellSize, int bx) {
    // map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float* pixel = div + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {

		float xc = x+0.5;
		float yc = y+0.5;
		float zc = z+0.5;

		float4 vL = tex3D(texVel,xc-1,yc,zc);
		float4 vR = tex3D(texVel,xc+1,yc,zc);
		float4 vT = tex3D(texVel,xc,yc+1,zc);
		float4 vB = tex3D(texVel,xc,yc-1,zc);
		float4 vBa = tex3D(texVel,xc,yc,zc-1);
		float4 vF = tex3D(texVel,xc,yc,zc+1);

		//obstacles
		float4 oL = tex3D(texObstacles,xc-1,yc,zc);
		float4 oR = tex3D(texObstacles,xc+1,yc,zc);
		float4 oT = tex3D(texObstacles,xc,yc+1,zc);
		float4 oB = tex3D(texObstacles,xc,yc-1,zc);
		float4 oBa = tex3D(texObstacles,xc,yc,zc-1);
		float4 oF = tex3D(texObstacles,xc,yc,zc+1);

		// Use obstacle velocities for solid cells:
		if (oL.w>0) vL = make_float4(oL.x,oL.y,oL.z,0);
		if (oR.w>0) vR = make_float4(oR.x,oR.y,oR.z,0);
		if (oT.w>0) vT = make_float4(oT.x,oT.y,oT.z,0);
		if (oB.w>0) vB = make_float4(oB.x,oB.y,oB.z,0);
		if (oBa.w>0) vT = make_float4(oBa.x,oBa.y,oBa.z,0);
		if (oF.w>0) vB = make_float4(oF.x,oF.y,oF.z,0);

		*pixel = 0.5 * (invCellSize.x*(vR.x - vL.x) + invCellSize.y*(vT.y - vB.y) + invCellSize.z*(vF.z-vBa.z));

	}
}

__global__ void jacobi(float *pressure, float alpha, float rBeta, cudaExtent gres, int bx) {
    // map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float* pixel = pressure + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {

		float xc = x+0.5;
		float yc = y+0.5;
		float zc = z+0.5;

		float pL = tex3D(texPressure,xc-1,yc,zc);
		float pR = tex3D(texPressure,xc+1,yc,zc);
		float pT = tex3D(texPressure,xc,yc+1,zc);
		float pB = tex3D(texPressure,xc,yc-1,zc);
		float pBa = tex3D(texPressure,xc,yc,zc-1);
		float pF = tex3D(texPressure,xc,yc,zc+1);

		float pC = tex3D(texPressure,xc,yc,zc);

		//obstacles
		float4 oL = tex3D(texObstacles,xc-1,yc,zc);
		float4 oR = tex3D(texObstacles,xc+1,yc,zc);
		float4 oT = tex3D(texObstacles,xc,yc+1,zc);
		float4 oB = tex3D(texObstacles,xc,yc-1,zc);
		float4 oBa = tex3D(texObstacles,xc,yc,zc-1);
		float4 oF = tex3D(texObstacles,xc,yc,zc+1);

		// Use center pressure for solid cells:
		if (oL.w>0) pL = pC;
		if (oR.w>0) pR = pC;
		if (oT.w>0) pT = pC;
		if (oB.w>0) pB = pC;
		if (oBa.w>0) pBa = pC;
		if (oF.w>0) pF = pC;


		float dC = tex3D(texDiv,xc,yc,zc);

		*pixel = (pL + pR + pB + pT + pBa + pF + alpha * dC) * rBeta;

	}
}

__global__ void projection(float4 *vel, cudaExtent gres, float3 invCellSize, int bx) {
    // map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + (blockIdx.x % bx) * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + (blockIdx.x / bx) * blockDim.x;

	// get a pointer to this pixel
	float4* pixel = vel + z*gres.width*gres.height + y*gres.width + x;

	if (x<gres.width && y<gres.height && z<gres.depth) {

		float xc = x+0.5;
		float yc = y+0.5;
		float zc = z+0.5;

		float pL = tex3D(texPressure,xc-1,yc,zc);
		float pR = tex3D(texPressure,xc+1,yc,zc);
		float pT = tex3D(texPressure,xc,yc+1,zc);
		float pB = tex3D(texPressure,xc,yc-1,zc);
		float pBa = tex3D(texPressure,xc,yc,zc-1);
		float pF = tex3D(texPressure,xc,yc,zc+1);


		float pC = tex3D(texPressure,xc,yc,zc);

		//obstacles
		float4 oL = tex3D(texObstacles,xc-1,yc,zc);
		float4 oR = tex3D(texObstacles,xc+1,yc,zc);
		float4 oT = tex3D(texObstacles,xc,yc+1,zc);
		float4 oB = tex3D(texObstacles,xc,yc-1,zc);
		float4 oBa = tex3D(texObstacles,xc,yc,zc-1);
		float4 oF = tex3D(texObstacles,xc,yc,zc+1);

		float4 obstV = make_float4(0,0,0,0);
		float4 vMask = make_float4(1,1,1,1);

		if (oT.w > 0) { pT = pC; obstV.y = oT.y; vMask.y = 0; }
		if (oB.w > 0) { pB = pC; obstV.y = oB.y; vMask.y = 0; }
		if (oR.w > 0) { pR = pC; obstV.x = oR.x; vMask.x = 0; }
		if (oL.w > 0) { pL = pC; obstV.x = oL.x; vMask.x = 0; }
		if (oBa.w > 0) { pBa = pC; obstV.z = oBa.z; vMask.z = 0; }
		if (oF.w > 0) { pF = pC; obstV.z = oF.z; vMask.z = 0; }

		float3 grad = 0.5*make_float3(invCellSize.x*(pR-pL), invCellSize.y*(pT-pB), invCellSize.z*(pF-pBa));

		float4 vNew = tex3D(texVel,xc,yc,zc) - make_float4(grad.x,grad.y,grad.z,0);

		*pixel = vMask * vNew + obstV;
		//*pixel = vNew;


	}
}

__global__ void displaySliceTex(float4 *optr, cudaExtent gres, float slice) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * gres.width;

	if (x<gres.width && y<gres.height) {

		optr[offset].x=optr[offset].y=optr[offset].z=optr[offset].w = tex3D(texDens,x+0.5,y+0.5,slice+0.5);
	}

}

__global__ void displayVectorSliceTex(float4 *optr, cudaExtent gres, float slice) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * gres.width;

	if (x<gres.width && y<gres.height) {

		float4 vel = tex3D(texVel,x+0.5,y+0.5,slice+0.5);

		optr[offset].x = vel.x;
		optr[offset].y = vel.y;
		optr[offset].z = vel.z;
		optr[offset].w = 1.0;
	}

}

__device__ float linstep(float val, float minval, float maxval) {

	return clamp((val-minval)/(maxval-minval), -1.0f, 1.0f);

}

__global__ void displayScalarSlice(float4 *optr, float* scalar, cudaExtent gres, float slice,
								   int sliceAxis, float minBound, float maxBound) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = 0;
	float res = 0;
	float* pixel;

	if (sliceAxis == 2) {
		if (x<gres.width && y<gres.height) {
			offset = x + y * gres.width;
			pixel = scalar + (int)slice*gres.width*gres.height + y*gres.width + x;
			res = *pixel;
		}
	} else if (sliceAxis == 0) {
		if (x<gres.depth && y<gres.height) {
			offset = x + y * gres.depth;
			pixel = scalar + x*gres.width*gres.height + y*gres.width + (int)slice;
			res = *pixel;
		}
	} else {
		if (x<gres.width && y<gres.depth) {
			offset = x + y * gres.width;
			pixel = scalar + y*gres.width*gres.height + (int)slice*gres.width + x;
			res = *pixel;
		}
	}

		optr[offset].x=optr[offset].y=optr[offset].z=optr[offset].w = linstep(res,minBound, maxBound);
	

}

__global__ void displayObstacles(float4 *optr, float4* vector, cudaExtent gres, float slice,
								   int sliceAxis) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = 0;
	float4 res = make_float4(0,0,0,0);
	float4* pixel;

	if (sliceAxis == 2) {
		if (x<gres.width && y<gres.height) {
			offset = x + y * gres.width;
			pixel = vector + (int)slice*gres.width*gres.height + y*gres.width + x;
			res = *pixel;
		}
	} else if (sliceAxis == 0) {
		if (x<gres.depth && y<gres.height) {
			offset = x + y * gres.depth;
			pixel = vector + x*gres.width*gres.height + y*gres.width + (int)slice;
			res = *pixel;
		}
	} else {
		if (x<gres.width && y<gres.depth) {
			offset = x + y * gres.width;
			pixel = vector + y*gres.width*gres.height + (int)slice*gres.width + x;
			res = *pixel;
		}
	}

		optr[offset].x=optr[offset].y=optr[offset].z=optr[offset].w = res.w;
	

}

__global__ void displayVectorSlice(float4 *optr, float4* vector, cudaExtent gres, float slice, int sliceAxis, float sliceBounds) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = 0;
	float4 res = make_float4(0,0,0,0);
	float4* pixel;

	if (sliceAxis == 2) {
		if (x<gres.width && y<gres.height) {
			offset = x + y * gres.width;
			pixel = vector + (int)slice*gres.width*gres.height + y*gres.width + x;
			res = *pixel;
		}
	} else if (sliceAxis == 0) {
		if (x<gres.depth && y<gres.height) {
			offset = x + y * gres.depth;
			pixel = vector + x*gres.width*gres.height + y*gres.width + (int)slice;
			res = *pixel;
		}

	} else {
		if (x<gres.width && y<gres.depth) {
			offset = x + y * gres.width;
			pixel = vector + y*gres.width*gres.height + (int)slice*gres.width + x;
			res = *pixel;
		}
	}


		optr[offset].x = linstep(res.x,-sliceBounds,sliceBounds);
		optr[offset].y = linstep(res.y,-sliceBounds,sliceBounds);
		optr[offset].z = linstep(res.z,-sliceBounds,sliceBounds);
		optr[offset].w = 1.0;

}

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
    
		FILE* fichier = NULL;
		
		fichier = fopen("C:\\error\\cudaerror.txt", "w");
		
		if (fichier != NULL) {
	 			fprintf(fichier, "%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
			fclose(fichier);
		}

        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 


void computeGridSize(cudaExtent res, dim3 &blocks, dim3 &threads, int &bx){

	int nThreads = 8;

	threads.x = nThreads;
	threads.y = nThreads;
	threads.z = nThreads;
	
	blocks.x = (res.width/nThreads + (!(res.width%nThreads)?0:1))
					* (res.depth/nThreads + (!(res.depth%nThreads)?0:1));
	blocks.y = res.height/nThreads + (!(res.height%nThreads)?0:1);
	
	blocks.z = 1;
	
	bx = ceil((float)res.width/(float)nThreads);

}

extern "C" void createBorderCu(float4* dev_obstacles,cudaExtent res,
								int borderPosX, int borderNegX,
								int borderPosY, int borderNegY,
								int borderPosZ, int borderNegZ ){
	dim3 threads, blocks;
	int bx;
	computeGridSize(res, blocks, threads, bx);			
			
	createBorder<<<blocks,threads>>>(dev_obstacles, res, borderPosX, borderNegX,
														borderPosY, borderNegY,
														borderPosZ, borderNegZ, bx);
															
}

extern "C" void addColliderCu(float4* dev_obstacles, float radius, float3 position,
								cudaExtent res, float3 vel){
	
	dim3 threads, blocks;
	int bx;
	computeGridSize(res, blocks, threads, bx);	
	
	addCollider<<<blocks,threads>>>(dev_obstacles, radius, position, res, bx, vel);
												
}

extern "C" void advectVelCu(float4* dev_vel, float timestep, float velDamp, float3 invGridSize,
							cudaExtent res){
																		
	dim3 threads, blocks;
	int bx;
	computeGridSize(res, blocks, threads, bx);
	
	advectVel<<<blocks,threads>>>(dev_vel,timestep,velDamp,invGridSize,res, bx);																			
}

extern "C" void advectDensCu(float* dev_dens,float timestep, float densDis, float3 invGridSize,
							cudaExtent res){

	dim3 threads, blocks;
	int bx;
	computeGridSize(res, blocks, threads, bx);		
	
	advectDens<<<blocks,threads>>>(dev_dens,timestep,densDis,invGridSize,res, bx);					
}

extern "C" void addDensCu(float* dev_dens, float timestep, float radius, float amount, float3 position,
						  cudaExtent res){
						  
	dim3 threads, blocks;
	int bx;
	computeGridSize(res, blocks, threads, bx);	
	
	addDens<<<blocks,threads>>>(dev_dens,timestep,radius,amount,position,res,bx);					  
}	

extern "C" void addDensBuoyCu(float4* dev_vel, float timestep, float densBuoyStrength, float3 densBuoyDir,
							  cudaExtent res){
							  
	dim3 threads, blocks;
	int bx;
	computeGridSize(res, blocks, threads, bx);	
	
	addDensBuoy<<<blocks,threads>>>(dev_vel,timestep,densBuoyStrength,densBuoyDir,res, bx);						 					  
}

extern "C" void calcNoiseCu(float* dev_noise, cudaExtent res, float3 fluidSize, float yOffset,
									int noiseOct, float noiseLacun, float gain, float noiseFreq, float noiseAmp) {


	dim3 threads, blocks;
	int bx;
	computeGridSize(res, blocks, threads, bx);

	float xDelta = fluidSize.x/(float)res.width;
	float yDelta = fluidSize.y/(float)res.height;
	float zDelta = fluidSize.z/(float)res.depth;


	cudaMemcpyToSymbol(c_perm_3d, h_perm, sizeof(h_perm),0,cudaMemcpyHostToDevice );

	k_perlin<<< blocks, threads>>>(dev_noise, res, make_float3(xDelta, yDelta, zDelta), yOffset,
									noiseOct, noiseLacun, 0.75f, noiseFreq, noiseAmp, bx);
}

extern "C" void addNoiseCu(float4* dev_vel, float timestep, float noiseStr, cudaExtent res){

	dim3 threads, blocks;
	int bx;
	computeGridSize(res, blocks, threads, bx);
	
	addNoise<<<blocks,threads>>>(dev_vel, timestep, noiseStr, res, bx);
}

extern "C" void vorticityCu(float4* dev_vort, cudaExtent res, float3 invCellSize){

	dim3 threads, blocks;
	int bx;
	computeGridSize(res, blocks, threads, bx);
	
	vorticity<<<blocks,threads>>>(dev_vort,res, invCellSize, bx);
}

extern "C" void vortConfCu(float4* dev_vel, float timestep, float vortConfStr, cudaExtent res){

	dim3 threads, blocks;
	int bx;
	computeGridSize(res, blocks, threads, bx);
	
	vortConf<<<blocks,threads>>>(dev_vel, timestep, vortConfStr, res, bx);
}

extern "C" void divergenceCu(float* dev_div, cudaExtent res, float3 invCellSize){

	dim3 threads, blocks;
	int bx;
	computeGridSize(res, blocks, threads, bx);
	
	divergence<<<blocks,threads>>>(dev_div,res,invCellSize, bx);
};

extern "C" void jacobiCu(float* dev_pressure, float alpha, float rBeta, cudaExtent res){

	dim3 threads, blocks;
	int bx;
	computeGridSize(res, blocks, threads, bx);
	
	jacobi<<<blocks,threads>>>(dev_pressure, alpha, rBeta, res, bx);
}

extern "C" void projectionCu(float4* dev_vel, cudaExtent res, float3 invCellSize){

	dim3 threads, blocks;
	int bx;
	computeGridSize(res, blocks, threads, bx);
	
	projection<<<blocks,threads>>>(dev_vel,res,invCellSize, bx);

}

extern "C" void setupTextures(){

	cudaChannelFormatDesc descFloat4_3d = cudaCreateChannelDesc<float4>();
	cudaChannelFormatDesc descFloat_3d = cudaCreateChannelDesc<float>();

	texDens.filterMode = cudaFilterModeLinear;
	texNoise.filterMode = cudaFilterModeLinear;
	texVel.filterMode = cudaFilterModeLinear;
	texDiv.filterMode = cudaFilterModeLinear;
	texPressure.filterMode = cudaFilterModeLinear;
	texObstacles.filterMode = cudaFilterModeLinear;
	texVort.filterMode = cudaFilterModeLinear;

}

extern "C" void bindDensTex(cudaArray* densArray){

	cudaChannelFormatDesc descFloat_3d = cudaCreateChannelDesc<float>();
	HANDLE_ERROR(cudaBindTextureToArray(texDens, densArray, descFloat_3d));

}

extern "C" void bindNoiseTex(cudaArray* noiseArray){

	cudaChannelFormatDesc descFloat_3d = cudaCreateChannelDesc<float>();
	HANDLE_ERROR(cudaBindTextureToArray(texNoise, noiseArray, descFloat_3d));

}

extern "C" void bindVelTex(cudaArray* velArray){

	cudaChannelFormatDesc descFloat_3d = cudaCreateChannelDesc<float4>();
	HANDLE_ERROR(cudaBindTextureToArray(texVel, velArray, descFloat_3d));

}

extern "C" void bindDivTex(cudaArray* divArray){

	cudaChannelFormatDesc descFloat_3d = cudaCreateChannelDesc<float>();
	HANDLE_ERROR(cudaBindTextureToArray(texDiv, divArray, descFloat_3d));

}

extern "C" void bindPressureTex(cudaArray* pressureArray){

	cudaChannelFormatDesc descFloat_3d = cudaCreateChannelDesc<float>();
	HANDLE_ERROR(cudaBindTextureToArray(texPressure, pressureArray, descFloat_3d));

}

extern "C" void bindObstaclesTex(cudaArray* obstaclesArray){

	cudaChannelFormatDesc descFloat_3d = cudaCreateChannelDesc<float4>();
	HANDLE_ERROR(cudaBindTextureToArray(texObstacles, obstaclesArray, descFloat_3d));

}

extern "C" void bindVortTex(cudaArray* vortArray){

	cudaChannelFormatDesc descFloat_3d = cudaCreateChannelDesc<float4>();
	HANDLE_ERROR(cudaBindTextureToArray(texVort, vortArray, descFloat_3d));

}

extern "C" void unbindDensTex(){
	cudaUnbindTexture(texDens);
}

extern "C" void unbindNoiseTex(){
	cudaUnbindTexture(texNoise);
}

extern "C" void unbindVelTex(){
	cudaUnbindTexture(texVel);
}

extern "C" void unbindDivTex(){
	cudaUnbindTexture(texDiv);
}

extern "C" void unbindPressureTex(){
	cudaUnbindTexture(texPressure);
}

extern "C" void unbindObstaclesTex(){
	cudaUnbindTexture(texObstacles);
}

extern "C" void unbindVortTex(){
	cudaUnbindTexture(texVort);
}

extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix) {
   cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix,0,cudaMemcpyHostToDevice );
}

extern "C" void renderFluidCu(float4 *d_output, uint imageW, uint imageH, float opaScale, cudaExtent res,
							float focalLength, float3 fluidSize, float3 lightPos,
							float stepMul, float shadowStepMul, float shadowThres, float shadowDens, int doShadows) {

	dim3 blockSize(16, 16);
	dim3 gridSize(imageW / blockSize.x, imageH / blockSize.y);
	
	float3 bottomLeft = make_float3(-0.5*fluidSize.x,-0.5*fluidSize.y,-0.5*fluidSize.z);
	float3 upperRight = -bottomLeft;
	float3 invSize = make_float3(1/fluidSize.x,1/fluidSize.y,1/fluidSize.z);
	float maxSize = max(max(fluidSize.x,fluidSize.y),fluidSize.z);

	if(doShadows) {
		d_render_shadows<<<gridSize, blockSize>>>( d_output, imageW, imageH, opaScale, res,
		focalLength, bottomLeft, upperRight, invSize, fluidSize, maxSize,
		lightPos, stepMul, shadowStepMul, shadowThres, shadowDens);
		
	} else {
		d_render<<<gridSize, blockSize>>>( d_output, imageW, imageH, opaScale, res,
		focalLength, bottomLeft, upperRight, invSize, maxSize, stepMul);
	}


}

extern "C" void renderFluidSliceCu(float4* d_output, cudaExtent res, float slicePos, int sliceAxis, int sliceType, float sliceBounds,
									float* dev_dens, float4* dev_vel, float* dev_noise, float* dev_pressure,
									float4* dev_vort, float4* dev_obstacles) {
	int nthreads = 16;

	dim3	blocks;
    dim3    threads(nthreads,nthreads);

	float slicePosUnNorm;

	if (sliceAxis == 2) {
		slicePosUnNorm = res.depth*slicePos;
		blocks.x = res.width/nthreads + (!(res.width%nthreads)?0:1);
		blocks.y = res.height/nthreads + (!(res.height%nthreads)?0:1);
	} else if(sliceAxis == 0) {
		slicePosUnNorm = res.width*slicePos;
		blocks.x = res.depth/nthreads + (!(res.depth%nthreads)?0:1);
		blocks.y = res.height/nthreads + (!(res.height%nthreads)?0:1);
	} else {
		slicePosUnNorm = res.height*slicePos;
		blocks.x = res.width/nthreads + (!(res.width%nthreads)?0:1);
		blocks.y = res.depth/nthreads + (!(res.depth%nthreads)?0:1);
	}

	/*eAttr.addField("Density",0);
	eAttr.addField("Velocity",1);
	eAttr.addField("Noise",2);
	eAttr.addField("Pressure",3);
	eAttr.addField("Vorticity",4);
	5 obstacles
	*/

	if (sliceType==0) {
		displayScalarSlice<<< blocks, threads>>>(d_output, dev_dens, res, slicePosUnNorm, sliceAxis, 0, sliceBounds);
	} else if (sliceType==1) {
		displayVectorSlice<<< blocks, threads>>>(d_output, dev_vel, res, slicePosUnNorm, sliceAxis, sliceBounds);
	} else if (sliceType==2) {
		displayScalarSlice<<< blocks, threads>>>(d_output, dev_noise, res, slicePosUnNorm, sliceAxis, -sliceBounds, sliceBounds);
	} else if (sliceType==3) {
		displayScalarSlice<<< blocks, threads>>>(d_output, dev_pressure, res, slicePosUnNorm, sliceAxis, -sliceBounds, sliceBounds);
	} else if (sliceType==4) {
		displayVectorSlice<<< blocks, threads>>>(d_output, dev_vort, res, slicePosUnNorm, sliceAxis, sliceBounds);
	} else if (sliceType==5) {
		displayObstacles<<< blocks, threads>>>(d_output, dev_obstacles, res, slicePosUnNorm, sliceAxis);
	}


}

extern "C" void addFluidForceKernelCu(float3* vel, float3* pos, cudaExtent gres, float3 invSize, float strength,
									  float dt, int nParts){


	int nthreads = min(256, nParts);

	int nBlocks = nParts/nthreads + (!(nParts%nthreads)?0:1);

    dim3 blocks(nBlocks, 1,1);
    dim3 threads(nthreads, 1, 1);

	addFluidForceKernel<<<blocks,threads>>>(vel, pos, gres, invSize, strength, dt, nParts);

}