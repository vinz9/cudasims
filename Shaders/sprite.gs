#version 120
#extension GL_EXT_geometry_shader4 : enable

uniform float pointRadius;  // point size in world space

void main() {

	float radius = pointRadius;

    // eye space
    vec3 pos = gl_PositionIn[0].xyz;
 
    vec3 x = vec3(radius, 0.0, 0.0);
    vec3 y = vec3(0.0, -radius, 0.0);

    // output quad
    gl_FrontColor = gl_FrontColorIn[0];
    gl_TexCoord[0] = vec4(0, 0, 0, 0);
    gl_TexCoord[1] = gl_PositionIn[0];
    gl_Position = gl_ProjectionMatrix * vec4(pos + x + y, 1);
    EmitVertex();

	gl_FrontColor = gl_FrontColorIn[0];
    gl_TexCoord[0] = vec4(0, 1, 0, 0);
     gl_TexCoord[1] = gl_PositionIn[0];
    gl_Position = gl_ProjectionMatrix * vec4(pos + x - y, 1);
    EmitVertex();

	gl_FrontColor = gl_FrontColorIn[0];
    gl_TexCoord[0] = vec4(1, 0, 0, 0);
     gl_TexCoord[1] = gl_PositionIn[0];
    gl_Position = gl_ProjectionMatrix * vec4(pos - x + y, 1);
    EmitVertex();

	gl_FrontColor = gl_FrontColorIn[0];
    gl_TexCoord[0] = vec4(1, 1, 0, 0);
     gl_TexCoord[1] = gl_PositionIn[0];
    gl_Position = gl_ProjectionMatrix * vec4(pos - x - y, 1);
    EmitVertex();
    
}