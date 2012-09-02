#include <stdio.h>
#include <GL/glew.h>
#include "GL/glut.h"

namespace cu{
	#include <cuda_runtime_api.h>

	#include <cuda_gl_interop.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>

}

#include "../../CudaCommon/CudaFluidSolver2D/vhFluidSolver.h"

int pause = 0;
unsigned int timer = 0;



// static method used for glut callbacks
static void idle_func( void ) {
    static int ticks = 1;

    glutPostRedisplay();
}

// static method used for glut callbacks
static void Key(unsigned char key, int x, int y) {

	VHFluidSolver* fluidSolver = VHFluidSolver::solverList[0];

    switch (key) {
        case 27:

            exit(0);
		 case 'r':
			 fluidSolver->resetFluid();
			 break;
		  case 'p':
			 if(pause==0)
				 pause = 1;
			 else
				pause = 0;
			 break;
    }
}

static void reshapeFunc(int w, int h)
{
    VHFluidSolver* fluidSolver = VHFluidSolver::solverList[0];
	float sizeX = fluidSolver->fluidSize.x*0.5;
	float sizeY = fluidSolver->fluidSize.y*0.5;
	
	glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
	glOrtho(-sizeX, sizeX, -sizeY, sizeY, 0.0, 1.0);

}

void computeFPS()
{

    char fps[256];
	float ifps = 1.f / (cu::cutGetAverageTimerValue(timer) / 1000.f);
    sprintf(fps, "2D Fluid: %3.1f fps", ifps);  

    glutSetWindowTitle(fps);

	cu::cutResetTimer(timer);  

}

// static method used for glut callbacks
static void Draw( void ) {

	cu::cutStartTimer(timer);  

    glClearColor( 0.0, 0.0, 0.0, 1.0 );
    glClear( GL_COLOR_BUFFER_BIT );

	VHFluidSolver* fluidSolver = VHFluidSolver::solverList[0];

	if (pause == 0)
		fluidSolver->solveFluid();

	//glDisable(GL_DEPTH_TEST);

	fluidSolver->drawFluid(0,0,0,0,0,0);

	//glEnable(GL_DEPTH_TEST);

    glutSwapBuffers();
	//glutReportErrors();

	cu::cutStopTimer(timer);  

    computeFPS();
}

int main( void ) {

	int c=1;
    char* dummy = "";
	
	int width = 512;
	int height = 512;

    glutInit( &c, &dummy );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( width, height );
	//glutInitWindowPosition(100, 100);
    glutCreateWindow( "bitmap" );

	glewInit();
	cu::cutilSafeCall(cu::cudaGLSetGLDevice( cu::cutGetMaxGflopsDeviceId() ));

	int dim = 200;

	VHFluidSolver* fluidSolver = new VHFluidSolver();

	fluidSolver->initFluidSolver(dim,dim);
	fluidSolver->resetFluid();

    glutKeyboardFunc(Key);
    glutDisplayFunc(Draw);
   // if (clickDrag != NULL)
   //     glutMouseFunc( mouse_func );
    glutIdleFunc( idle_func );
	glutReshapeFunc(reshapeFunc);


	cu::cutCreateTimer( &timer);

    glutMainLoop();

	delete fluidSolver;

	cu::cudaThreadExit();
}