#include <windows.h>
#include <stdio.h>
#include <GL/glew.h>
#include "../../Utils/nvModel.h"
#include "../../Utils/GLSLProgram.h"

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

int useFluid = 0;

int id1 = 0;

nv::Model *model;
nv::vec3f modelMin, modelMax;

GLSLProgram         *simpleProg;

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


	//glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(viewTranslation.x, viewTranslation.y, viewTranslation.z);
	glRotatef(viewRotation.y, 0.0, 1.0, 0.0);
    glRotatef(viewRotation.x, 1.0, 0.0, 0.0);

	glVertexPointer( model->getPositionSize(),
		GL_FLOAT, model->getCompiledVertexSize() * sizeof(float),
		model->getCompiledVertices());

    glNormalPointer( GL_FLOAT, model->getCompiledVertexSize() * sizeof(float),
		model->getCompiledVertices() + model->getCompiledNormalOffset());


	glEnableClientState( GL_VERTEX_ARRAY);
    glEnableClientState( GL_NORMAL_ARRAY);

    //CHECK_ERRORS;

	simpleProg->enable();

	int tada = model->getCompiledIndexCount( nv::Model::eptTriangles);

    glDrawElements( GL_TRIANGLES, model->getCompiledIndexCount( nv::Model::eptTriangles),
		GL_UNSIGNED_INT, model->getCompiledIndices( nv::Model::eptTriangles));

	simpleProg->disable();

    glDisableClientState( GL_VERTEX_ARRAY);
    glDisableClientState( GL_NORMAL_ARRAY);

	
  
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

	glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);								// The Type Of Depth Testing To Do

	model = new nv::Model;

	std::string modelPath = "C:/Users/vinz/Documents/vhDev/VHCudaTools/torus.obj";

	if (model->loadModelFromFile(modelPath.c_str())) {

            // remove any primitives with duplicate indices, as they might confuse adjacency
            //model->removeDegeneratePrims();

            // compute normal and tengent space
            model->computeNormals();
           // model->computeTangents();

            // make the model efficient for rendering with vertex arrays
            //model->compileModel( nv::Model::eptAll);

			model->compileModel( nv::Model::eptTriangles);

            // get the bounding box to help place the model on-screen
            model->computeBoundingBox( modelMin, modelMax);
        } else {
            //fprintf(stderr, "Error loading model '%s'\n", model_filename);
            delete model;
            model = 0;
        }

	simpleProg = new GLSLProgram("regular_vertex.glsl", "regular_fragment.glsl");

	glutMainLoop();


	cu::cudaThreadExit();

}