#version 450

#extension GL_EXT_nonuniform_qualifier : require

#include "../utility/material.glslh"

// TransparentPass: forward-shades glTF BLEND surfaces with the same lights as PbrPass, writing
// premultiplied colour that the pass blends back to front (src + dst * (1 - src.a)) into its own
// layer. Opaque faces drawn by the same meshes are discarded — GeometryPass already handled them.

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUv;

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

layout(set = 0, binding = 1) uniform sampler2D diffuseTex[];
layout(set = 0, binding = 2) uniform sampler2D normalTex[];
layout(set = 0, binding = 3) uniform sampler2D metallicRoughnessTex[];
layout(set = 0, binding = 4) uniform sampler2D emissiveTex[];
layout(set = 0, binding = 5) readonly buffer FaceGroupIndices
{
    uint values[];
} faceGroupIndices;
layout(set = 0, binding = 6) readonly buffer Materials
{
    MaterialData data[];
} materials;
// Bindings 7-10 are geometry.vert's skinning buffers.

layout(set = 0, binding = 12) uniform samplerCube irradianceMap;
layout(set = 0, binding = 13) uniform samplerCube prefilterMap;
layout(set = 0, binding = 14) uniform sampler2D brdfLut;
layout(set = 0, binding = 15) uniform sampler2D ltc1;
layout(set = 0, binding = 16) uniform sampler2D ltc2;

#include "../utility/lighting.glslh"

layout(set = 0, binding = 11) readonly buffer LightBuffer
{
    LightData lights[];
};

layout(set = 0, binding = 17) uniform sampler2DArrayShadow spotShadowMap;
layout(set = 0, binding = 18) uniform SpotShadowData
{
    mat4 lightViewProj[16];
    vec4 pcss[16];
    uvec4 header;
} spotShadow;
layout(set = 0, binding = 19) uniform sampler2DArrayShadow cascadedShadowMap;
layout(set = 0, binding = 20) uniform CascadedShadowData
{
    mat4 lightViewProj[4];
    vec4 splitDepths;
    vec4 pcss[4];
    uvec4 header;
} cascadedShadow;
// See pbr.frag: the blocker search reads stored depth, so the maps are bound again uncompared.
layout(set = 0, binding = 21) uniform sampler2DArray spotShadowMapDepth;
layout(set = 0, binding = 22) uniform sampler2DArray cascadedShadowMapDepth;
// See pbr.frag: area light shadows, compared and uncompared.
layout(set = 0, binding = 23) uniform sampler2DArrayShadow areaShadowMap;
layout(set = 0, binding = 24) uniform AreaShadowData
{
    mat4 lightViewProj[8];
    vec4 pcss[8];
    uvec4 header;
} areaShadow;
layout(set = 0, binding = 25) uniform sampler2DArray areaShadowMapDepth;
layout(set = 0, binding = 26) uniform sampler2DArrayShadow pointShadowMap;
layout(set = 0, binding = 27) uniform PointShadowData
{
    mat4 lightViewProj[24];
    vec4 pcss[24];
    uvec4 header;
} pointShadow;
layout(set = 0, binding = 28) uniform sampler2DArray pointShadowMapDepth;

#include "../utility/shadow_sampling.glslh"

// The first four members mirror geometry.vert's block (see geometry.frag for primitiveIdOffset).
layout(push_constant) uniform PC
{
    mat4 model;
    uint primitiveIdOffset;
    uint paletteOffset;
    uint skinEnabled;
    uint numLights;
    uint pfMips;
} pc;

layout(location = 0) out vec4 outColor;

void main()
{
    uint faceGroupIndex = faceGroupIndices.values[pc.primitiveIdOffset + gl_PrimitiveID];
    MaterialData mat = materials.data[faceGroupIndex];
    if (!isFlagEnabled(mat.alphaBlend) || (!gl_FrontFacing && !isFlagEnabled(mat.doubleSided)))
    {
        discard;
    }

    vec4 baseColor = texture(diffuseTex[nonuniformEXT(faceGroupIndex)], inUv) * mat.baseColorFactor;
    if (baseColor.a <= 0.0)
    {
        discard;
    }
    float roughness =
        max(texture(metallicRoughnessTex[nonuniformEXT(faceGroupIndex)], inUv).g * mat.roughnessFactor, 0.045);
    float metallic = texture(metallicRoughnessTex[nonuniformEXT(faceGroupIndex)], inUv).b * mat.metallicFactor;
    vec3 emissive = texture(emissiveTex[nonuniformEXT(faceGroupIndex)], inUv).rgb * mat.emissiveFactor.rgb;

    // Normal mapping, as geometry.frag does it, but kept in view space for the lighting functions.
    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent.xyz);
    if (!gl_FrontFacing)
    {
        N = -N;
    }
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T) * inTangent.w;
    vec3 tangentNormal = texture(normalTex[nonuniformEXT(faceGroupIndex)], inUv).rgb * 2.0 - 1.0;
    // Kept in world space as well: the shadow lookups need it there for their receiver-plane threshold.
    vec3 worldNormal = normalize(mat3(T, B, N) * tangentNormal);
    vec3 normal = normalize(mat3(cameraUbo.view) * worldNormal);
    vec3 position = (cameraUbo.view * vec4(inWorldPos, 1.0)).xyz;

    // HBAO only sees the G-buffer, so transparent surfaces are unoccluded (ao = 0).
    vec3 color = emissive;
    for (uint i = 0; i < pc.numLights; ++i)
    {
        // 0: this pass shades once per fragment, so there is no second evaluation to decorrelate.
        float visibility = lightShadowVisibility(lights[i], position, inWorldPos, worldNormal,
                                                 gl_FragCoord.xy, 0u);
        color += visibility * ShadeLight(lights[i], position, normal, baseColor.rgb, roughness, metallic, 0.0,
                                         pc.pfMips);
    }

    outColor = vec4(color * baseColor.a, baseColor.a);
}
