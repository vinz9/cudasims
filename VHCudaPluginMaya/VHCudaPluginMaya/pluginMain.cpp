#include <maya/MFnPlugin.h> 

#include "fluidNode2D.h"
#include "fluidNode3D.h"
#include "particlesNode.h"
#include "initGLCuda.h"

MStatus initializePlugin( MObject obj )
{ 
	MStatus   status;
	MFnPlugin plugin( obj, "Foliativ", "0.9", "Any");

	status = plugin.registerNode("vhFluidSolver2D", fluidNode2D::id, fluidNode2D::creator,
		fluidNode2D::initialize, MPxNode::kLocatorNode);

	status = plugin.registerNode("vhFluidSolver3D", fluidNode3D::id, fluidNode3D::creator,
		fluidNode3D::initialize, MPxNode::kLocatorNode);

	status = plugin.registerNode("vhParticles", particlesNode::id, particlesNode::creator,
		particlesNode::initialize, MPxNode::kLocatorNode);

	status = plugin.registerNode("initGLCuda", initGLCuda::id, initGLCuda::creator,
		initGLCuda::initialize);
	
	return status;
}

MStatus uninitializePlugin( MObject obj )
{
	MStatus   status;
	MFnPlugin plugin( obj );

	status = plugin.deregisterNode(fluidNode2D::id);

	status = plugin.deregisterNode(fluidNode3D::id);

	status = plugin.deregisterNode(particlesNode::id);

	status = plugin.deregisterNode(initGLCuda::id);

	return status;
}