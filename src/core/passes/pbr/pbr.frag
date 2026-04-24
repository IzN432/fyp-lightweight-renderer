#version 450

#include "../utility/geometry.glslh"

layout(set = 0, binding = 0) uniform CameraUbo
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    vec4 cameraPosition;
} cameraUbo;

layout(set = 0, binding = 1) uniform samplerCube irradianceMap;
layout(set = 0, binding = 2) uniform samplerCube prefilterMap;
layout(set = 0, binding = 3) uniform sampler2D gbufferDepth;
layout(set = 0, binding = 4) uniform sampler2D gbufferAlbedo;
layout(set = 0, binding = 5) uniform sampler2D gbufferNormal;
layout(set = 0, binding = 6) uniform sampler2D gbufferRoughnessMetallic;

layout(location = 0) out vec4 outColor;

void main()
{
    vec2 uvs = gl_FragCoord.xy / vec2(textureSize(gbufferAlbedo, 0));
    outColor = vec4(texture(gbufferAlbedo, uvs).rgb, 1.0);
}