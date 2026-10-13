#version 450

// Instanced particle spheres. One unit-radius sphere mesh is drawn once per particle; the
// per-instance centre and radius come from a vertex buffer rather than push constants so the
// whole fluid is a single draw call.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inCenterAndRadius; // per-instance

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

layout(location = 0) out vec3 outNormal;

layout(push_constant) uniform PC
{
    vec4 albedo;
    float visualScale;
} pc;

void main()
{
    const vec3 worldPos = (inCenterAndRadius.xyz + inPosition * inCenterAndRadius.w) * pc.visualScale;
    outNormal = normalize(inNormal);
    gl_Position = cameraUbo.viewProj * vec4(worldPos, 1.0);
}
