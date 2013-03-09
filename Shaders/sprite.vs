void main(void) {

	gl_FrontColor = gl_Color; 
	
    gl_Position = gl_ModelViewMatrix * gl_Vertex;

}