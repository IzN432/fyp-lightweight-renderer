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
    vec4 pcss[16];
    uvec4 header;
} spotShadow;
layout(set = 0, binding = 15) uniform sampler2DArrayShadow cascadedShadowMap;
layout(set = 0, binding = 16) uniform CascadedShadowData
{
    mat4 lightViewProj[4];
    vec4 splitDepths;
    vec4 pcss[4];
    uvec4 header;
} cascadedShadow;
// The same two images again without the comparison sampler: PCSS's blocker search needs the stored
// depth, which a sampler2DArrayShadow can never hand back.
layout(set = 0, binding = 17) uniform sampler2DArray spotShadowMapDepth;
layout(set = 0, binding = 18) uniform sampler2DArray cascadedShadowMapDepth;
// Area light shadows: one layer per quad face (see AreaShadowPass), plus the uncompared view for
// the blocker search.
layout(set = 0, binding = 19) uniform sampler2DArrayShadow areaShadowMap;
layout(set = 0, binding = 20) uniform AreaShadowData
{
    mat4 lightViewProj[8];
    vec4 pcss[8];
    uvec4 header;
} areaShadow;
layout(set = 0, binding = 21) uniform sampler2DArray areaShadowMapDepth;

#include "../utility/shadow_sampling.glslh"

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

    // The shadow lookups need the normal in world space for their receiver-plane threshold. The view
    // matrix is rigid, so its upper 3x3 rotates normals correctly without an inverse transpose.
    vec3 worldNormal = normalize(mat3(cameraUbo.invView) * normal);

    vec3 color = vec3(0.0);
    for (uint i = 0; i < pc.numLights; ++i)
    {
        vec3 worldPosition = (cameraUbo.invView * vec4(position, 1.0)).xyz;
        // sampleIndex decorrelates the shadow dither between this pixel's coverage samples, which
        // main() averages — without it all of them rotate the filter the same way.
        float visibility = lightShadowVisibility(lights[i], position, worldPosition, worldNormal,
                                                 gl_FragCoord.xy, uint(sampleIndex));
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
