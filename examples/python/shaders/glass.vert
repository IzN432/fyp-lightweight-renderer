#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(set = 0, binding = 0) uniform Camera
{
    mat4 view;
    mat4 proj;
} camera;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor    = inColor;
    gl_Position = camera.proj * camera.view * vec4(inPosition, 1.0);
}
