Provided as is with no warranty.


!!! This a tech demo/toy more than a production ready tool !!!

For both versions, Shaders need to be in C:\Shaders (this is hardcoded, sorry), sprite default path is C:\pictures\centerGradient.tif (provided) but can be changed.


Maya plugin :

------------------------------------------------
windows install :

copy VHFluidCuda2013X64.mll in a folder called plug-ins in your maya documents folder. 

Might not be needed : 
copy the files in the dll folder in the bin directory of your maya installation (where maya.exe is located)

copy *Template.mel in your scripts directory

------------------------------------------------
usage :

check the mel procedures in fluidSolverSetup.mel to get you started.


fluid2DSetup(string $prefix) sets up a 2D fluid container with 2 sphere emitters and 1 sphere collider.
fluid3DSetup(string $prefix) sets up a 3D fluid container with 1 sphere emitter and 1 sphere collider.
setupCudaParticles(string $prefix) sets up a particle System with 1 sphere emitter.
setupFluidDrivenParts(string $prefix) sets up a 3D fluid container with 1 sphere emitter, 1 sphere collider and a particle system with one emitter.

Fluid :

Fluid colliders are locators with a radius attribute which controls the size of the sphere.
Fluid emitters are maya fluids emitters, but besides there translate and rotate, only the max distance and Density/Voxel/Sec are taken into account for now.

The container can be translated and rotated, but scale has no effect (except on the wire cube).

The transform cSolverEmptyTr is there to force the solver to update every frame when maya is playing, if you hide it the solver won't update.
A maya fluid container is set up with the correct connections and hidden by the script.
If you check copyDens (uncheck Preview for a 2D fluid), at each frame the density is copied from the cuda solver to the maya container, which will allow you to shade it (with the settings on the maya fluid node) and render it.
But it is much slower than if the data stays on the GPU between simulation and display.

Most parameters are quite similar to those of a maya Fluid or self explanatory.
For a 3D fluid, the fluid is raymarched in a texture. Display Res defines the size of the texture. The bigger the texture, the more rays are cast (one per pixel), the slower the rendering, especially with shadows on. Shadows are done with a brute force method and are quite slow. Shadow step mul is a multiplier on the size of the base step to speed things up.

Particles :

Particles emitters are locators with a radius attribute and an amount attribute.
The particles are only living on the GPU and can't be rendered except by doing a playblast for now.


-------------------------------------------------

Houdini plugin :

------------------------------------------------
windows install :

copy SOP_vhCudaSims.dll in a folder called dso in your Houdini11/12 documents folder.
I quickly hacked the Houdini 12 version, HDK and viewport have changed a bit, particles/sprites display work using H11 viewport but fluid doesn't display.
Might not be needed : 
copy the files in the dll folder in the bin directory of your Houdini installation (where hmaster.exe is located)


------------------------------------------------
usage :

check the example files to get you started.

Fluid :

emitters are points with a radius and an amount attribute connected to the first input.
colliders are points with a radius attribute connected to the second input.

The position and rotation on the fluidsolver sop are taken into account with the emitters and colliders positions.
You can transform the volume below in the network to move it, or at obj level, this will not change the sim.


The previews work correctly if you disable material shaders on the effects tab in the display options (d shortcut on the viewport)
If you uncheck the Preview and DisplaySlice (only Preview for a 2D fluid), at each frame the density is copied from the cuda solver to the houdini volume, which will allow you to and render it with Mantra.
But it is much slower than if the data stays on the GPU between simulation and display.
For a 3d fluid, you can copy the velocity field to houdini volumes, and use them to advect particles. It is also slower though.
Be careful though, you need to disable multithreading on the advect pop or it will crash!

Most parameters are self explanatory.
For a 3D fluid, the fluid is raymarched in a texture. Display Res defines the size of the texture. The bigger the texture, the more rays are cast (one per pixel), the slower the rendering, especially with shadows on. Shadows are done with a brute force method and are quite slow. Shadow step mul is a multiplier on the size of the base step to speed things up.

Particles :

Particles emitters are points with a radius attribute and an amount attribute, and a couple more attribute for the velocity, check the sample.
The particles can be copied to Houdini points (when preview is unchecked) but it's very slow so I didn't finish this yet (no colour nor opacity).

Now particles can have trails, and there is a new attractor force, check the examples.

--------------------------------------------------

Not much error checking is done, but the plugin shouldn't crash too much.

Tested successfully with a geforce 460 GTX.
Advection with the 3D solver doesn't work on older cards with compute capability < 1.3, I will fix it in a later release.
Not sure if this happens with other geforces

To build the source code you need the GPU computing SDK 3.2 x64 and the cuda toolkit 4.0 x64.


The standalone versions were for quick testing and have no GUI...

11/09/2012 : version 0.7 : release with shadowed sprites. Houdini version has trail and attractors which are not exposed in Maya version.
Known bug : Particles advection doesn't work properly if the fluid container is not a the world's origin. Fluid emitters and colliders are transformed in the fluid space, but that's not the case for the particles.
Maya version crashes if viewport is switched between shaded/wireframe and shadowed sprites are displayed.
Probably last version.

22/07/2011 : version 0.61 : first linux build!
04/05/2011 : version 0.6 : basic particles system with fluid advection added, code refactoring
29/03/2011 : version 0.55 : initial houdini public release, code refactoring
04/03/2011 : version 0.5 : initial public release.

To do:
More complex emitters
Collisions with meshes
Fields
Better advection and algorithms


Thanks to Rob Farber for the perlin noise cuda code and NVIDIA for the shadowed particles sample.


Please credit me and drop me a line if you do something cool with it!

vincent.houze@foliativ.net
http://www.foliativ.net

