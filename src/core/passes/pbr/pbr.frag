#version 450

#include "../utility/geometry.glslh"

layout (location = 0) in vec2 inUV;

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

layout(set = 0, binding = 1) uniform samplerCube irradianceMap;
layout(set = 0, binding = 2) uniform samplerCube prefilterMap;
layout(set = 0, binding = 3) uniform sampler2D brdfLut;
layout(set = 0, binding = 4) uniform sampler2D ltc1; // inverse M for area light cosine warp
layout(set = 0, binding = 5) uniform sampler2D ltc2; // GGX norm, fresnel, unused, horizon-clip

#include "../utility/lighting.glslh"

// These are the unresolved images from GeometryPass. sampler2DMS deliberately has no filtering or
// normalized coordinates: texelFetch selects one exact coverage sample from the current pixel.
layout(set = 0, binding = 6) uniform sampler2DMS gbufferDepth;
layout(set = 0, binding = 7) uniform sampler2DMS gbufferAlbedo;
layout(set = 0, binding = 8) uniform sampler2DMS gbufferNormal;
layout(set = 0, binding = 9) uniform sampler2DMS gbufferRoughnessMetallic;
layout(set = 0, binding = 10) uniform sampler2DMS gbufferEmissive;
layout(set = 0, binding = 11) readonly buffer LightBuffer
{
    LightData lights[];
};
layout(set = 0, binding = 12) uniform sampler2D ao;
layout(set = 0, binding = 13) uniform sampler2DArrayShadow spotShadowMap;
layout(set = 0, binding = 14) uniform SpotShadowData
{
    mat4 lightViewProj[16];
    uvec4 header;
} spotShadow;

layout(push_constant) uniform PC {
    uint pfMips;
    uint numLights;
} pc;

layout(location = 0) out vec4 outColor;

vec3 ShadeSample(ivec2 pixel, int sampleIndex, out bool covered)
{
    // Fetch a coherent set of attributes from the same coverage sample. Resolving attributes first
    // would mix foreground material values with clear values at silhouettes and then light the mix.
    vec3 albedo = texelFetch(gbufferAlbedo, pixel, sampleIndex).rgb;
    vec3 normal = unpackViewNormal(texelFetch(gbufferNormal, pixel, sampleIndex).rg);
    vec2 material = texelFetch(gbufferRoughnessMetallic, pixel, sampleIndex).rg;
    float roughness = max(material.r, 0.045);
    float metallic = material.g;
    float depth = texelFetch(gbufferDepth, pixel, sampleIndex).r;

    if (depth >= 1.0) {
        covered = false;
        return vec3(0.0);
    }
    covered = true;

    vec3 position = depthToViewPosition(depth, inUV, cameraUbo.invProj);
    float occlusion = texture(ao, inUV).r;

    vec3 color = vec3(0.0);
    for (uint i = 0; i < pc.numLights; ++i)
    {
        float visibility = 1.0;
        int shadowIndex = lights[i].shadowIndex < spotShadow.header.x ? int(lights[i].shadowIndex) : -1;
        if (shadowIndex >= 0 && lights[i].type == LIGHT_TYPE_SPOT)
        {
            vec3 worldPosition = (cameraUbo.invView * vec4(position, 1.0)).xyz;
            vec4 lightClip = spotShadow.lightViewProj[shadowIndex] * vec4(worldPosition, 1.0);
            vec3 shadowCoord = lightClip.xyz / lightClip.w;
            vec2 uv = shadowCoord.xy * 0.5 + 0.5;
            if (shadowCoord.z >= 0.0 && shadowCoord.z <= 1.0 && all(greaterThanEqual(uv, vec2(0.0))) &&
                all(lessThanEqual(uv, vec2(1.0))))
            {
                vec2 texel = 1.0 / vec2(textureSize(spotShadowMap, 0).xy);
                visibility = 0.0;
                for (int y = -1; y <= 1; ++y)
                    for (int x = -1; x <= 1; ++x)
                        visibility += texture(spotShadowMap,
                                              vec4(uv + vec2(x, y) * texel, float(shadowIndex),
                                                   shadowCoord.z - 0.0005));
                visibility /= 9.0;
            }
        }
        color += visibility * ShadeLight(lights[i], position, normal, albedo, roughness, metallic, occlusion, pc.pfMips);
    }

    // Emissive is a direct, unlit contribution (e.g. the visual quad representing an area light) —
    // it doesn't go through the light loop above.
    color += texelFetch(gbufferEmissive, pixel, sampleIndex).rgb;

    return color;
}

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    int sampleCount = textureSamples(gbufferDepth);
    vec3 accumulated = vec3(0.0);
    int coveredSamples = 0;

    // Shade every covered sample, then average only the lit samples. Coverage is carried in alpha so
    // the final/composite pass can blend the shaded surface with the background at polygon edges.
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
    {
        bool covered;
        vec3 sampleColor = ShadeSample(pixel, sampleIndex, covered);
        if (covered)
        {
            accumulated += sampleColor;
            ++coveredSamples;
        }
    }

    float coverage = float(coveredSamples) / float(sampleCount);
    vec3 color = coveredSamples > 0 ? accumulated / float(coveredSamples) : vec3(0.0);
    outColor = vec4(color, coverage);
}
