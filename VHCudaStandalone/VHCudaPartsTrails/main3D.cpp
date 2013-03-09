#include <windows.h>
#include <stdio.h>
#include <GL/glew.h>


#include "../../CudaCommon/CudaParticlesSystem/vhParticlesSystem.h"
#include "../../CudaCommon/CudaFluidSolver3D/vhFluidSolver3D.h"

namespace cu {
	#include <cuda_runtime_api.h>
	#include <cuda_gl_interop.h>

	#include <vector_functions.h>
	#include <cutil.h>
	#include <cutil_inline_runtime.h>
}

#include "GL/glext.h"
#include "GL/glut.h"

void computeFPS();

int vwidth = 803;
int vheight = 713;
float fov = 35;
//float fov = 60;

unsigned int timer = 0;

cu::float3 viewRotation= cu::make_float3(1, 0, 0);
cu::float3 viewTranslation = cu::make_float3(0, 0, -8.0f);

VHParticlesSystem* particlesSystem = NULL;
VHFluidSolver3D* fluidSolver = NULL;

int useFluid = 0;

int id1 = 0;

void idle_func( void ) {
    static int ticks = 1;
    glutPostRedisplay();
}

void Key(unsigned char key, int x, int y) {

    switch (key) {
		case 27:
            //exit(0);

		case 'r':
			
			 break;

		case 'p':
			
			 break;

		case 's':
			
			 break;
    }
}

void reshapeFunc(int w, int h) {

    glViewport(0, 0, w, h);

	vwidth = w;
	vheight = h;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	gluPerspective(fov,(float)w/(float)h,1,1000);

	glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y){

    if (state == GLUT_DOWN)
        buttonState  |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;
    glutPostRedisplay();
}

void motion(int x, int y){

    float dx, dy;
    dx = x - ox;
    dy = y - oy;

    if (buttonState == 3) {
        // left+middle = zoom
        viewTranslation.z += dy / 100.0;
    } 
    else if (buttonState & 2) {
        // middle = translate
        viewTranslation.x += dx / 100.0;
        viewTranslation.y -= dy / 100.0;
    }
    else if (buttonState & 1) {
        // left = rotate
        viewRotation.x += dy / 5.0;
        viewRotation.y += dx / 5.0;
    }

    ox = x; oy = y;
    glutPostRedisplay();
}

void display(){

	cu::cutStartTimer(timer);

	/*int maxDrawBuffers;
	glGetIntegerv ( GL_MAX_DRAW_BUFFERS_ARB, &maxDrawBuffers );

	int db;
	glGetIntegerv ( GL_DRAW_BUFFER, &db );

	int dbb;
	glGetIntegerv ( GL_DRAW_BUFFER0, &dbb );

	int dba;
	glGetIntegerv ( GL_AUX_BUFFERS, &dba );*/

	if(useFluid)
		fluidSolver->solveFluid();


	if (particlesSystem->preview) {
		// maxparts/life
		particlesSystem->emitParticles();

		particlesSystem->updateParticles();

		//((TurbulenceForce*)(particlesSystem->leadsForces[2]))->noiseOffset.y +=0.05;
		//((TurbulenceForce*)(particlesSystem->trailsForces[2]))->noiseOffset.y +=0.05;

		//((AttractorForce*)(particlesSystem->leadsForces[4]))->origin = cu::make_float3(sin(particlesSystem->time)*2,sin(particlesSystem->time*1)*1,sin(particlesSystem->time*0.5)*2);
		//((AttractorForce*)(particlesSystem->leadsForces[5]))->origin = cu::make_float3(cos(particlesSystem->time)*2,cos(particlesSystem->time*1)*1,sin(particlesSystem->time*0.5)*2);

		//((AttractorForce*)(particlesSystem->trailsForces[4]))->origin = cu::make_float3(sin(particlesSystem->time)*2,sin(particlesSystem->time*1)*1,sin(particlesSystem->time*0.5)*2);
		//((AttractorForce*)(particlesSystem->trailsForces[5]))->origin = cu::make_float3(cos(particlesSystem->time)*2,cos(particlesSystem->time*1)*1,sin(particlesSystem->time*0.5)*2);
	
	}


	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(viewTranslation.x, viewTranslation.y, viewTranslation.z);
	glRotatef(viewRotation.y, 0.0, 1.0, 0.0);
    glRotatef(viewRotation.x, 1.0, 0.0, 0.0);

	/*TextureManager::Inst()->BindTexture(id1);

	int size = 2;

	glEnable(GL_TEXTURE_2D);

			glBegin( GL_QUADS );
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-size,-size,0.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-size,size,0.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex3f(size,size,0.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex3f(size,-size,0.0f);
		glEnd();

	glDisable(GL_TEXTURE_2D);*/

	//glutWireCube(5);

	if (particlesSystem->preview)
		particlesSystem->draw();


	if(fluidSolver->preview)
		fluidSolver->drawFluid(viewRotation.x,viewRotation.y,0,viewTranslation.x,viewTranslation.y,viewTranslation.z);

  
    glutSwapBuffers();
	//glutReportErrors();


	cu::cutStopTimer(timer);  

    computeFPS();
}

void computeFPS() {

    char fps[256];
	float ifps = 1.f / (cu::cutGetAverageTimerValue(timer) / 1000.f);
    sprintf(fps, "Parts: %3.1f fps", ifps);  

    glutSetWindowTitle(fps);

	cu::cutResetTimer(timer);  

}

//fluid alone
void preset1() {

	particlesSystem->preview = 0;

	fluidSolver->initFluidSolver(60,120,30);

	fluidSolver->fluidSize = cu::make_float3(4.0,8.0,2.0);
	fluidSolver->emitters[0].posY = -1.6; //-4

	fluidSolver->preview = 1;
	fluidSolver->opaScale = 0.2;


	fluidSolver->resetFluid();
	fluidSolver->solveFluid();

	useFluid = 1;

}

//parts alone
void preset2() {

	particlesSystem->initParticlesSystem(4000000,1);

	particlesSystem->emitters[0].amount = 200000;

	particlesSystem->preview = 1;
	particlesSystem->startColor = cu::make_float3(0.39,0.17,0.95);
	particlesSystem->endColor = cu::make_float3(1,0.97,0.33);

	((TurbulenceForce*)(particlesSystem->leadsForces[2]))->strength = 2;

}

//parts points in fluid
void preset3(){

	particlesSystem->initParticlesSystem(4000000,1);
	particlesSystem->emitters[0].amount = 200000;


	particlesSystem->emitters[0].pos = cu::make_float3(0,-1.5,0);
	particlesSystem->emitters[0].radVelAmp = 3.0;

	
	((FluidForce*)(particlesSystem->leadsForces[3]))->strength = 1.0;
	((DampingForce*)(particlesSystem->leadsForces[0]))->strength = 30;

	particlesSystem->opacity = 1;
	particlesSystem->preview = 1;
	particlesSystem->startColor = cu::make_float3(0.39,0.17,0.95);
	particlesSystem->endColor = cu::make_float3(1,0.97,0.33);


	fluidSolver->initFluidSolver(50,50,50);

	fluidSolver->fluidSize = cu::make_float3(5.0,5.0,5.0);
	fluidSolver->emitters[0].posY = -1.5; //-4

	fluidSolver->preview = 0;
	fluidSolver->opaScale = 0.2;


	fluidSolver->resetFluid();

	((FluidForce*)(particlesSystem->leadsForces[3]))->fluidSolver = fluidSolver;

	useFluid = 1;

}
//parts sprites in fluid
void preset3a(){

	//particlesSystem->initParticlesSystem(4000000,1);
	//particlesSystem->emitters[0].amount = 200000;

	//particlesSystem->path = "C:/pictures/clouds.tif";

	particlesSystem->initParticlesSystem(200000,1);
	particlesSystem->emitters[0].amount = 10000;

	particlesSystem->emitters[0].pos = cu::make_float3(0,-1.5,0);
	particlesSystem->emitters[0].radVelAmp = 3.0;

	
	((FluidForce*)(particlesSystem->leadsForces[3]))->strength = 1.0;
	((DampingForce*)(particlesSystem->leadsForces[0]))->strength = 30;

	particlesSystem->opacity = 1;
	particlesSystem->pRend->pointSize = 0.03;
	particlesSystem->pRend->displayMode = VHParticlesRender::SPRITES;
	particlesSystem->preview = 1;
	particlesSystem->startColor = cu::make_float3(0.39,0.17,0.95);
	particlesSystem->endColor = cu::make_float3(1,0.97,0.33);


	fluidSolver->initFluidSolver(50,50,50);

	fluidSolver->fluidSize = cu::make_float3(5.0,5.0,5.0);
	fluidSolver->emitters[0].posY = -1.5; //-4

	fluidSolver->preview = 0;
	fluidSolver->opaScale = 0.2;


	fluidSolver->resetFluid();

	((FluidForce*)(particlesSystem->leadsForces[3]))->fluidSolver = fluidSolver;

	useFluid = 1;

}

//parts sprites in fluid
void preset3b(){

	//particlesSystem->initParticlesSystem(4000000,1);
	//particlesSystem->emitters[0].amount = 200000;

	//particlesSystem->path = "C:/pictures/clouds.tif";

	particlesSystem->initParticlesSystem(100000,1);
	particlesSystem->emitters[0].amount = 5000;

	particlesSystem->emitters[0].pos = cu::make_float3(0,-1.5,0);
	particlesSystem->emitters[0].radVelAmp = 3.0;

	
	((FluidForce*)(particlesSystem->leadsForces[3]))->strength = 1.0;
	((DampingForce*)(particlesSystem->leadsForces[0]))->strength = 30;

	particlesSystem->opacity = 1;
	particlesSystem->pRend->pointSize = 0.03;
	particlesSystem->pRend->displayMode = VHParticlesRender::SPRITES;
	particlesSystem->preview = 1;
	//particlesSystem->startColor = cu::make_float3(0.39,0.17,0.95);
	//particlesSystem->endColor = cu::make_float3(1,0.97,0.33);


	fluidSolver->initFluidSolver(50,50,50);

	fluidSolver->fluidSize = cu::make_float3(5.0,5.0,5.0);
	fluidSolver->emitters[0].posY = -1.5; //-4

	fluidSolver->preview = 0;
	fluidSolver->opaScale = 0.2;


	fluidSolver->resetFluid();

	((FluidForce*)(particlesSystem->leadsForces[3]))->fluidSolver = fluidSolver;

	useFluid = 1;

}

//parts sprites in fluid
void preset3c(){

	//particlesSystem->initParticlesSystem(4000000,1);
	//particlesSystem->emitters[0].amount = 200000;

	particlesSystem->initParticlesSystem(500000,1);

	particlesSystem->pRend->loadSprite("C:/pictures/centerGradient.tif");

	particlesSystem->emitters[0].amount = 100000;

	particlesSystem->emitters[0].radius = 0.25;
	particlesSystem->emitters[0].pos = cu::make_float3(0,-1.5,0);
	//particlesSystem->emitters[0].vel = cu::make_float3(1,0,0);
	particlesSystem->emitters[0].radVelAmp = 0.0;

	
	((FluidForce*)(particlesSystem->leadsForces[3]))->strength = 1.0;
	((DampingForce*)(particlesSystem->leadsForces[0]))->strength = 30;

	particlesSystem->opacity = 0.1;

	/*particlesSystem->pRend->pointSize = 1;*/
	particlesSystem->pRend->pointSize = 0.025;
	particlesSystem->pRend->displayMode = VHParticlesRender::SHADOWED_SPRITES;

	particlesSystem->pRend->sortParts = 1;

	//particlesSystem->pRend->displayMode = VHParticlesRender::POINTS;

	particlesSystem->preview = 1;
	/*particlesSystem->startColor = cu::make_float3(0.39,0.17,0.95);
	particlesSystem->endColor = cu::make_float3(1,0.97,0.33);*/

	((GravityForce*)(particlesSystem->leadsForces[1]))->strength = 1.0;


	fluidSolver->initFluidSolver(50,50,50);

	fluidSolver->fluidSize = cu::make_float3(5.0,5.0,5.0);
	fluidSolver->emitters[0].posY = -1.5; //-4

	fluidSolver->preview = 0;
	fluidSolver->opaScale = 0.2;


	fluidSolver->resetFluid();

	((FluidForce*)(particlesSystem->leadsForces[3]))->fluidSolver = fluidSolver;

	useFluid = 1;

}

void preset3d(){

	//particlesSystem->initParticlesSystem(4000000,1);
	//particlesSystem->emitters[0].amount = 200000;

	particlesSystem->pRend->loadSprite("C:/pictures/clouds.tif");

	particlesSystem->initParticlesSystem(500000,1);
	particlesSystem->emitters[0].amount = 100000;

	particlesSystem->emitters[0].radius = 0.25;
	particlesSystem->emitters[0].pos = cu::make_float3(0,-1.5,0);
	//particlesSystem->emitters[0].vel = cu::make_float3(1,0,0);
	particlesSystem->emitters[0].radVelAmp = 0.0;

	
	((FluidForce*)(particlesSystem->leadsForces[3]))->strength = 1.0;
	((DampingForce*)(particlesSystem->leadsForces[0]))->strength = 30;

	particlesSystem->opacity = 1;

	//particlesSystem->pRend->pointSize = 1;
	particlesSystem->pRend->pointSize = 0.025;
	particlesSystem->pRend->displayMode = VHParticlesRender::SPRITES;
	//particlesSystem->pRend->displayMode = VHParticlesRender::POINTS;

	particlesSystem->preview = 0;
	particlesSystem->startColor = cu::make_float3(0.39,0.17,0.95);
	particlesSystem->endColor = cu::make_float3(1,0.97,0.33);

	((GravityForce*)(particlesSystem->leadsForces[1]))->strength = 1.0;


	fluidSolver->initFluidSolver(50,50,50);

	fluidSolver->fluidSize = cu::make_float3(5.0,5.0,5.0);
	fluidSolver->emitters[0].posY = -1.5; //-4

	fluidSolver->preview = 0;
	fluidSolver->opaScale = 0.2;


	fluidSolver->resetFluid();
	fluidSolver->preview = 1;

	((FluidForce*)(particlesSystem->leadsForces[3]))->fluidSolver = fluidSolver;

	useFluid = 1;

}

//trail turbulence
void preset4() {

	particlesSystem->initParticlesSystem(30000,80);

	particlesSystem->emitters[0].amount = 3000;
	particlesSystem->emitters[0].pos = cu::make_float3(0,-1.5,0);

	particlesSystem->inheritVel = 0.5;

	particlesSystem->opacity = 0.1;
	particlesSystem->preview = 1;

	particlesSystem->partsLife = 5;
	particlesSystem->partsLifeVar = 0;

	((TurbulenceForce*)(particlesSystem->leadsForces[2]))->strength = 2;
	((TurbulenceForce*)(particlesSystem->trailsForces[2]))->strength = 1;

	((GravityForce*)(particlesSystem->leadsForces[1]))->strength = 0.0;


	((GravityForce*)(particlesSystem->trailsForces[1]))->strength = 0.5;
	((GravityForce*)(particlesSystem->trailsForces[1]))->gravityDir = cu::make_float3(0,-1,0);

	particlesSystem->startColor = cu::make_float3(0.39,0.17,0.95);
	particlesSystem->endColor = cu::make_float3(1,0.97,0.33);
		particlesSystem->pRend->displayMode = VHParticlesRender::LINES;



}

//trails fluid
void preset5() {

	particlesSystem->initParticlesSystem(20000,100);

	particlesSystem->emitters[0].amount = 500;
	particlesSystem->emitters[0].pos = cu::make_float3(0,-1.5,0);
	particlesSystem->emitters[0].radVelAmp = 0.1;

	particlesSystem->inheritVel = 0.0;

	particlesSystem->opacity = 0.1;


	particlesSystem->partsLife = 5;
	particlesSystem->partsLifeVar = 0;

	((DampingForce*)(particlesSystem->leadsForces[0]))->strength = 30;
	((DampingForce*)(particlesSystem->trailsForces[0]))->strength = 40;

	((FluidForce*)(particlesSystem->leadsForces[3]))->strength = 1.0;
	((FluidForce*)(particlesSystem->trailsForces[3]))->strength = 1.0;

	particlesSystem->startColor = cu::make_float3(0.39,0.17,0.95);
	particlesSystem->endColor = cu::make_float3(1,0.97,0.33);
	particlesSystem->pRend->displayMode = VHParticlesRender::LINES;
	particlesSystem->preview = 1;


	fluidSolver->initFluidSolver(40,40,40);

	fluidSolver->fluidSize = cu::make_float3(5.0,5.0,5.0);
	fluidSolver->emitters[0].posY = -1.5; //-4

	fluidSolver->preview = 0;

	fluidSolver->resetFluid();

	((FluidForce*)(particlesSystem->leadsForces[3]))->fluidSolver = fluidSolver;
	((FluidForce*)(particlesSystem->trailsForces[3]))->fluidSolver = fluidSolver;

	useFluid = 1;
}

void preset6() {

	particlesSystem->initParticlesSystem(2000000,1);
	particlesSystem->emitters[0].amount = 200000;

	/*particlesSystem->initParticlesSystem(40000,50);
	particlesSystem->drawLines = 01;
	particlesSystem->inheritVel = 0.3;
	particlesSystem->emitters[0].amount = 2000;*/

	particlesSystem->emitters[0].radius = 0.5;
	particlesSystem->emitters[0].pos = cu::make_float3(0,0,0);
	//particlesSystem->emitters[0].radVelAmp = 3.0;

	
	((DampingForce*)(particlesSystem->leadsForces[0]))->strength = 1;
	((DampingForce*)(particlesSystem->trailsForces[0]))->strength = 10;

	((TurbulenceForce*)(particlesSystem->leadsForces[2]))->strength = 2;
	((TurbulenceForce*)(particlesSystem->trailsForces[2]))->strength = 2;

	((GravityForce*)(particlesSystem->leadsForces[1]))->strength = 1.0;
	((GravityForce*)(particlesSystem->leadsForces[1]))->gravityDir = cu::make_float3(0,-1,0);

	particlesSystem->leadsForces[4] = new AttractorForce(particlesSystem);
	((AttractorForce*)(particlesSystem->leadsForces[4]))->strength = -2.0;
	((AttractorForce*)(particlesSystem->leadsForces[4]))->decay = 1;

	particlesSystem->leadsForces[5] = new AttractorForce(particlesSystem);
	((AttractorForce*)(particlesSystem->leadsForces[5]))->strength = -2.0;
	((AttractorForce*)(particlesSystem->leadsForces[5]))->decay = 1;

	((AttractorForce*)(particlesSystem->leadsForces[4]))->origin = cu::make_float3(1,1,0);
	((AttractorForce*)(particlesSystem->leadsForces[5]))->origin = cu::make_float3(-1,1,0);

	particlesSystem->nLeadsForces = 6;

	particlesSystem->trailsForces[4] = new AttractorForce(particlesSystem);
	((AttractorForce*)(particlesSystem->trailsForces[4]))->strength = -2.0;
	((AttractorForce*)(particlesSystem->trailsForces[4]))->decay = 1;

	particlesSystem->trailsForces[5] = new AttractorForce(particlesSystem);
	((AttractorForce*)(particlesSystem->trailsForces[5]))->strength = -2.0;
	((AttractorForce*)(particlesSystem->trailsForces[5]))->decay = 1;

	particlesSystem->nTrailsForces = 5;

	//((AttractorForce*)(particlesSystem->leadsForces[4]))->origin = cu::make_float3(0,1,0);

	particlesSystem->preview = 1;
	//particlesSystem->opacity = 1.0;
	particlesSystem->startColor = cu::make_float3(0.39,0.17,0.95);
	particlesSystem->endColor = cu::make_float3(1,0.97,0.33);

}


int main( void ) {

	int c=1;
    char* dummy = "";

    glutInit( &c, &dummy );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( vwidth, vheight );
    glutCreateWindow( "parts" );

	glewInit();
	cu::cutilSafeCall(cu::cudaGLSetGLDevice(cu::cutGetMaxGflopsDeviceId() ));

	glutKeyboardFunc(Key);
	glutDisplayFunc(display);
	glutMouseFunc(mouse);
    glutIdleFunc( idle_func );
	glutReshapeFunc(reshapeFunc);
	glutMotionFunc(motion);

	cu::cutCreateTimer( &timer);


	particlesSystem = new VHParticlesSystem();
	fluidSolver = new VHFluidSolver3D();

	preset1();
	//preset2();

	//preset3();
	//preset3a();
	//preset3b();

	//preset3c();

	//preset3d();
	//preset4();
	//preset5();
	//preset6();

	//TextureManager::Inst()->LoadTexture("C:/pictures/clouds.dds", id1, GL_BGRA, GL_RGBA);

	glutMainLoop();


	delete particlesSystem;
	delete fluidSolver;

	cu::cudaThreadExit();

}