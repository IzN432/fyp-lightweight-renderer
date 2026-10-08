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

void main()
{
    uint faceGroupIndex = faceGroupIndices.values[pc.primitiveIdOffset + gl_PrimitiveID];
    MaterialData mat = materials.data[faceGroupIndex];
    float alpha = texture(diffuseTex[nonuniformEXT(faceGroupIndex)], inUv).a * mat.baseColorFactor.a;
    if (shouldDiscardOpaqueFragment(mat, alpha, gl_FrontFacing))
        discard;
}
