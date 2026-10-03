#version 450

layout(location = 0) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(0.05, 0.08, 0.12, 1.0);
}
