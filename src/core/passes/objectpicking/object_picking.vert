#version 450

#include "../utility/skinning_types.glslh"

layout(location = 0) in vec3 inPosition;
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

layout(std430, set = 0, binding = 7) readonly buffer SkinInfluenceEntries { SkinInfluence entries[]; } skinInfluences;
layout(std430, set = 0, binding = 8) readonly buffer SkinInfluenceOffsets { uint offsets[]; } skinInfluenceOffsets;
layout(std430, set = 0, binding = 9) readonly buffer SkinPositionIndices { uint indices[]; } skinPositionIndices;
layout(std430, set = 0, binding = 10) readonly buffer SkinJointMatrices { mat4 matrices[]; } skinJointMatrices;

#define LR_SKIN_INFLUENCES skinInfluences.entries
#define LR_SKIN_INFLUENCE_OFFSETS skinInfluenceOffsets.offsets
#define LR_SKIN_POSITION_INDICES skinPositionIndices.indices
#define LR_SKIN_JOINT_MATRICES skinJointMatrices.matrices
#include "../utility/skinning.glslh"

layout(location = 3) out vec2 outUv;

layout(push_constant) uniform PC
{
    mat4 model;
    uint primitiveIdOffset;
    uint paletteOffset;
    uint skinEnabled;
    uint pickingId;
} pc;

void main()
{
    vec3 position = inPosition;
    if (pc.skinEnabled != 0)
    {
        position = (calculateSkinMatrix(gl_VertexIndex, pc.paletteOffset) * vec4(position, 1.0)).xyz;
    }
    gl_Position = cameraUbo.viewProj * pc.model * vec4(position, 1.0);
    outUv = inUv;
}
