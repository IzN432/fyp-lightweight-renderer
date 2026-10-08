#version 450

#include "../utility/skinning_types.glslh"

layout(location = 0) in vec3 inPosition;
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

layout(std430, set = 0, binding = 7) readonly buffer SkinInfluenceEntries
{
    SkinInfluence entries[];
} skinInfluences;

layout(std430, set = 0, binding = 8) readonly buffer SkinInfluenceOffsets
{
    uint offsets[];
} skinInfluenceOffsets;

layout(std430, set = 0, binding = 9) readonly buffer SkinPositionIndices
{
    uint indices[];
} skinPositionIndices;

layout(std430, set = 0, binding = 10) readonly buffer SkinJointMatrices
{
    mat4 matrices[];
} skinJointMatrices;

#define LR_SKIN_INFLUENCES skinInfluences.entries
#define LR_SKIN_INFLUENCE_OFFSETS skinInfluenceOffsets.offsets
#define LR_SKIN_POSITION_INDICES skinPositionIndices.indices
#define LR_SKIN_JOINT_MATRICES skinJointMatrices.matrices
#include "../utility/skinning.glslh"

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outTangent;
layout(location = 3) out vec2 outUv;

// Mirrors the push_constant block in geometry.frag — model is this draw's mesh-to-world matrix
// (identity for meshes whose vertices are already baked into world space, e.g. area light quads).
// primitiveIdOffset is unused here but must stay in the layout so both stages agree on offsets.
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
    vec3 normal = inNormal;
    vec3 tangent = inTangent.xyz;

    if (pc.skinEnabled != 0)
    {
        mat4 skinMatrix = calculateSkinMatrix(gl_VertexIndex, pc.paletteOffset);
        position = (skinMatrix * vec4(position, 1.0)).xyz;
        normal = transpose(inverse(mat3(skinMatrix))) * normal;
        tangent = mat3(skinMatrix) * tangent;
    }

    vec4 worldPos = pc.model * vec4(position, 1.0);
    outWorldPos = worldPos.xyz;

    mat3 normalMatrix = transpose(inverse(mat3(pc.model)));
    outNormal = normalize(normalMatrix * normal);
    outTangent = vec4(normalize(mat3(pc.model) * tangent), inTangent.w);

    outUv = inUv;
    gl_Position = cameraUbo.viewProj * worldPos;
}
