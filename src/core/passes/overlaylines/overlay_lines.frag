#version 450

layout(location = 0) in vec3 inColor;
layout(location = 1) in float inVisibleOpacity;
layout(location = 2) in float inOccludedOpacity;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D gbufferDepth;

void main()
{
    float depth = gl_FragCoord.z;
    vec2 screenSize = vec2(textureSize(gbufferDepth, 0));
    vec2 uv = gl_FragCoord.xy / screenSize;
    float sceneDepth = texture(gbufferDepth, uv).r;
    bool isOccluded = depth > sceneDepth + 1e-4;

    outColor = vec4(inColor, isOccluded ? inOccludedOpacity : inVisibleOpacity);
}
