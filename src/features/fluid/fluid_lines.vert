#version 450

// The domain box, drawn as plain world-space line segments.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(set = 0, binding = 0) uniform CameraUbo
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invView;
    mat4 invProj;
    vec3 cameraPosition;
    float padding;
} cameraUbo;

layout(location = 0) out vec3 outColor;

layout(push_constant) uniform PC
{
    float visualScale;
} pc;

void main()
{
    outColor = inColor;
    gl_Position = cameraUbo.viewProj * vec4(inPosition * pc.visualScale, 1.0);
}
