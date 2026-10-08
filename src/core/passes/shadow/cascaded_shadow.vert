#version 450
#extension GL_ARB_shader_viewport_layer_array : require
#include "../utility/skinning_types.glslh"
layout(location = 0) in vec3 inPosition;
layout(location = 3) in vec2 inUv;
layout(location = 0) out vec2 outUv;
layout(set = 0, binding = 0) uniform ShadowData
{
    mat4 lightViewProj[4];
    vec4 splitDepths;
    vec4 pcss[4];
    uvec4 header;
} shadow;
layout(std430, set = 0, binding = 7) readonly buffer SkinInfluenceEntries { SkinInfluence entries[]; } skinInfluences;
layout(std430, set = 0, binding = 8) readonly buffer SkinInfluenceOffsets { uint offsets[]; } skinInfluenceOffsets;
layout(std430, set = 0, binding = 9) readonly buffer SkinPositionIndices { uint indices[]; } skinPositionIndices;
layout(std430, set = 0, binding = 10) readonly buffer SkinJointMatrices { mat4 matrices[]; } skinJointMatrices;
#define LR_SKIN_INFLUENCES skinInfluences.entries
#define LR_SKIN_INFLUENCE_OFFSETS skinInfluenceOffsets.offsets
#define LR_SKIN_POSITION_INDICES skinPositionIndices.indices
#define LR_SKIN_JOINT_MATRICES skinJointMatrices.matrices
#include "../utility/skinning.glslh"
layout(push_constant) uniform PC
{
    mat4 model;
    uint primitiveIdOffset;
    uint paletteOffset;
    uint skinEnabled;
} pc;
void main()
{
    vec3 position = inPosition;
    if (pc.skinEnabled != 0)
        position = (calculateSkinMatrix(gl_VertexIndex, pc.paletteOffset) * vec4(position, 1.0)).xyz;
    uint layer = gl_InstanceIndex;
    outUv = inUv;
    gl_Layer = int(layer);
    gl_Position = shadow.lightViewProj[layer] * pc.model * vec4(position, 1.0);
}
