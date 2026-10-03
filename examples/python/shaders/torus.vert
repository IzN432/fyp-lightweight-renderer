#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(set = 0, binding = 0) uniform Camera
{
    mat4 view;
    mat4 proj;
} camera;

layout(push_constant) uniform Model
{
    mat4 model;
} pc;

layout(location = 0) out vec3 outNormal;

void main()
{
    outNormal   = mat3(pc.model) * inNormal;
    gl_Position = camera.proj * camera.view * pc.model * vec4(inPosition, 1.0);
}
