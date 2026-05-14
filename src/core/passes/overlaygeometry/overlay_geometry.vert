#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

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

layout(push_constant) uniform PC
{
    mat4 model;
} pc;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;

void main()
{
    outWorldPos = (pc.model * vec4(inPosition, 1.0)).xyz;
    outNormal = normalize(mat3(pc.model) * inNormal);

    gl_Position = cameraUbo.viewProj * vec4(outWorldPos, 1.0);
}