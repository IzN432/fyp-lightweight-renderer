#version 450

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

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

layout(set = 0, binding = 1) uniform sampler2D gbufferDepth;

layout(push_constant) uniform PC {
    layout(offset = 64) vec3 overlayColor;
    layout(offset = 76) float occludedOpacity;
} pc;

void main()
{
    float depth = gl_FragCoord.z;
    vec2 screenSize = vec2(textureSize(gbufferDepth, 0));
    vec2 uv = gl_FragCoord.xy / screenSize;
    float sceneDepth = texture(gbufferDepth, uv).r;

    bool isOccluded = depth > sceneDepth + 1e-4;
    outColor = vec4(pc.overlayColor, isOccluded ? pc.occludedOpacity : 1.0);
}