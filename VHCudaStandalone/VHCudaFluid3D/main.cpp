#include <stdio.h>
#include <GL/glew.h>
#include "GL/glut.h"
#include <cmath>

#include "../../CudaCommon/CudaFluidSolver3D/vhFluidSolver3D.h"

namespace cu {

	#include <cuda_runtime_api.h>
	#include <driver_functions.h>
	#include <cuda_gl_interop.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>
}

typedef unsigned int uint;
typedef unsigned char uchar;

void computeFPS();
void drawWireCube(float x, float y, float z);

unsigned int timer = 0;

cu::float3 viewRotation= cu::make_float3(1, 0, 0);
cu::float3 viewTranslation = cu::make_float3(0, 0, -12.0f);

int pause = 0;

int vwidth = 512;
int vheight = 512;
float fov = 35;

void idle_func( void ) {
    static int ticks = 1;
    glutPostRedisplay();
}

void Key(unsigned char key, int x, int y) {

	VHFluidSolver3D* fluidSolver = VHFluidSolver3D::solverList[0];

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

		case 's':
			if(fluidSolver->displaySlice==0) {
				 fluidSolver->displaySlice = 1;
				fluidSolver->preview = 0;
			} else {
				fluidSolver->displaySlice = 0;
				fluidSolver->preview = 1;
			}
			 break;
    }
}

void reshapeFunc(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

	vwidth = w;
	vheight = h;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	gluPerspective(fov,(float)w/(float)h,1,1000);

}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState  |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
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

void display()
{
	cu::cutStartTimer(timer);  

	VHFluidSolver3D* fluidSolver = VHFluidSolver3D::solverList[0];

	if (pause == 0)
		fluidSolver->solveFluid();

	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(viewTranslation.x, viewTranslation.y, viewTranslation.z);
	glRotatef(viewRotation.y, 0.0, 1.0, 0.0);
    glRotatef(viewRotation.x, 1.0, 0.0, 0.0);

	float sizeX = fluidSolver->fluidSize.x*0.5;
	float sizeY = fluidSolver->fluidSize.y*0.5;
	float sizeZ = fluidSolver->fluidSize.z*0.5;

	drawWireCube(sizeX,sizeY,sizeZ);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	if (fluidSolver->preview == 1) {

		fluidSolver->drawFluid(viewRotation.x,viewRotation.y,0,viewTranslation.x,viewTranslation.y,viewTranslation.z);
	}

	if (fluidSolver->displaySlice == 1) {

		fluidSolver->drawFluidSlice(0,0,0,0,0,0);

	}

	glDisable(GL_BLEND);
  
    glutSwapBuffers();
	//glutReportErrors();

	cu::cutStopTimer(timer);  

    computeFPS();
}

void computeFPS()
{

    char fps[256];
	float ifps = 1.f / (cu::cutGetAverageTimerValue(timer) / 1000.f);
    sprintf(fps, "3D Fluid: %3.1f fps", ifps);  

    glutSetWindowTitle(fps);

	cu::cutResetTimer(timer);  

}

void drawWireCube(float x, float y, float z) {

	glBegin(GL_LINE_STRIP);									// Draw A Quad
		glVertex3f( x, y,-z);					// Top Right Of The Quad (Top)
		glVertex3f(-x, y,-z);					// Top Left Of The Quad (Top)
		glVertex3f(-x, y, z);					// Bottom Left Of The Quad (Top)
		glVertex3f( x, y, z);					// Bottom Right Of The Quad (Top)
		glVertex3f( x, y,-z);
	glEnd();
	
	glBegin(GL_LINE_STRIP);
		glVertex3f( x,-y, z);					// Top Right Of The Quad (Bottom)
		glVertex3f(-x,-y, z);					// Top Left Of The Quad (Bottom)
		glVertex3f(-x,-y,-z);					// Bottom Left Of The Quad (Bottom)
		glVertex3f( x,-y,-z);					// Bottom Right Of The Quad (Bottom)
		glVertex3f( x,-y, z);
	glEnd();
		
	glBegin(GL_LINES);
		glVertex3f(-x, y, z);					// Top Left Of The Quad (Front)
		glVertex3f(-x,-y, z);					// Bottom Left Of The Quad (Front)
		glVertex3f( x, y, z);					// Top Right Of The Quad (Front)
		glVertex3f( x,-y, z);					// Bottom Right Of The Quad (Front)
	
		glVertex3f(-x,-y,-z);					// Top Left Of The Quad (Back)
		glVertex3f(-x, y,-z);					// Bottom Left Of The Quad (Back)
		glVertex3f( x, y,-z);					// Bottom Right Of The Quad (Back)
		glVertex3f( x,-y,-z);					// Top Right Of The Quad (Back)
	glEnd();
}


int main( void ) {

	int c=1;
    char* dummy = "";

    glutInit( &c, &dummy );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( vwidth, vheight );
    glutCreateWindow( "3DFluid" );

	glewInit();
	cu::cutilSafeCall(cu::cudaGLSetGLDevice(cu::cutGetMaxGflopsDeviceId() ));

	VHFluidSolver3D* fluidSolver = new VHFluidSolver3D();

	fluidSolver->initFluidSolver(60,120,30);

	fluidSolver->fluidSize = cu::make_float3(4.0,8.0,2.0);
	fluidSolver->emitters[0].posY = -1.6; //-4

	fluidSolver->preview = 1;
	fluidSolver->opaScale = 0.2;


	fluidSolver->resetFluid();
	fluidSolver->solveFluid();
	

    glutKeyboardFunc(Key);
	glutDisplayFunc(display);
	glutMouseFunc(mouse);
    glutIdleFunc( idle_func );
	glutReshapeFunc(reshapeFunc);
	glutMotionFunc(motion);

	cu::cutCreateTimer( &timer);

    glutMainLoop();

	delete fluidSolver;


	cu::cudaThreadExit();
}