#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(set = 0, binding = 0) uniform CameraUbo
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    vec4 cameraPosition;
} cameraUbo;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outUv;

void main()
{
    vec4 worldPos = vec4(inPosition, 1.0);
    outWorldPos = worldPos.xyz;
    outNormal = normalize(inNormal);
    outUv = inUv;
    gl_Position = cameraUbo.viewProj * worldPos;
}
