#version 450

#extension GL_EXT_nonuniform_qualifier : require

#include "../utility/material.glslh"

layout(location = 3) in vec2 inUv;

layout(set = 0, binding = 1) uniform sampler2D diffuseTex[];
layout(set = 0, binding = 5) readonly buffer FaceGroupIndices { uint values[]; } faceGroupIndices;
layout(set = 0, binding = 6) readonly buffer Materials { MaterialData data[]; } materials;

layout(push_constant) uniform PC
{
    mat4 model;
    uint primitiveIdOffset;
    uint paletteOffset;
    uint skinEnabled;
    uint pickingId;
} pc;

layout(location = 0) out uint outPickingId;

void main()
{
    uint faceGroupIndex = faceGroupIndices.values[pc.primitiveIdOffset + gl_PrimitiveID];
    MaterialData material = materials.data[faceGroupIndex];
    float alpha = texture(diffuseTex[nonuniformEXT(faceGroupIndex)], inUv).a * material.baseColorFactor.a;
    // BLEND geometry remains pickable. Only holes cut by MASK and culled one-sided backs are absent.
    if ((isFlagEnabled(material.alphaBlend) && alpha <= 0.001) || alpha < material.alphaCutoff ||
        (!gl_FrontFacing && !isFlagEnabled(material.doubleSided)))
    {
        discard;
    }
    outPickingId = pc.pickingId;
}
