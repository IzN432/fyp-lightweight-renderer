#version 450

#extension GL_EXT_nonuniform_qualifier : require

#include "../utility/material.glslh"

layout(location = 0) in vec2 inUv;

layout(set = 0, binding = 1) uniform sampler2D diffuseTex[];
layout(set = 0, binding = 5) readonly buffer FaceGroupIndices
{
    uint values[];
} faceGroupIndices;
layout(set = 0, binding = 6) readonly buffer Materials
{
    MaterialData data[];
} materials;

layout(push_constant) uniform PC
{
    mat4 model;
    uint primitiveIdOffset;
} pc;

// Stable pseudo-random threshold per shadow-map texel. Including the array layer prevents the
// same pattern from being projected by every light, while keeping the result temporally stable.
float alphaHashThreshold(uvec3 coordinate)
{
    uint hash = coordinate.x * 0x8da6b343u;
    hash ^= coordinate.y * 0xd8163841u;
    hash ^= coordinate.z * 0xcb1ab31fu;
    hash ^= hash >> 16;
    hash *= 0x7feb352du;
    hash ^= hash >> 15;
    hash *= 0x846ca68bu;
    hash ^= hash >> 16;
    return float(hash >> 8) * (1.0 / 16777216.0);
}

void main()
{
    uint faceGroupIndex = faceGroupIndices.values[pc.primitiveIdOffset + gl_PrimitiveID];
    MaterialData mat = materials.data[faceGroupIndex];
    float alpha = texture(diffuseTex[nonuniformEXT(faceGroupIndex)], inUv).a * mat.baseColorFactor.a;

    if (!gl_FrontFacing && mat.doubleSided < 0.5)
        discard;

    if (mat.alphaBlend > 0.5)
    {
        uvec3 shadowTexel = uvec3(uvec2(gl_FragCoord.xy), uint(gl_Layer));
        if (alpha <= alphaHashThreshold(shadowTexel))
            discard;
    }
    else if (alpha < mat.alphaCutoff)
        discard;
}
