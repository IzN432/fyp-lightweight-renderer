#version 450

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

layout(location = 0) in vec4 inPositionAndVisibleOpacity;
layout(location = 1) in vec4 inColorAndOccludedOpacity;

layout(location = 0) out vec3 outColor;
layout(location = 1) out float outVisibleOpacity;
layout(location = 2) out float outOccludedOpacity;

void main()
{
    gl_Position        = cameraUbo.viewProj * vec4(inPositionAndVisibleOpacity.xyz, 1.0);
    outColor           = inColorAndOccludedOpacity.xyz;
    outVisibleOpacity  = inPositionAndVisibleOpacity.w;
    outOccludedOpacity = inColorAndOccludedOpacity.w;
}
