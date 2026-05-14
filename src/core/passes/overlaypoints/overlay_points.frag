#version 450

layout(location = 0) in vec3 inColor;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D gbufferDepth;

layout(push_constant) uniform PC {
    layout(offset = 68) float occludedOpacity;
} pc;

void main()
{
    vec2 screenSize = vec2(textureSize(gbufferDepth, 0));
    vec2 uv = gl_FragCoord.xy / screenSize;
    float sceneDepth = texture(gbufferDepth, uv).r;

    bool isOccluded = gl_FragCoord.z > sceneDepth + 1e-4;
    outColor = vec4(inColor, isOccluded ? pc.occludedOpacity : 1.0);
}
