#version 450

#include "../../core/passes/utility/skinning_types.glslh"

layout(location = 0) in vec3 inPosition;

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
#include "../../core/passes/utility/skinning.glslh"

// Mirrors SmokeVoxelizePC in SmokeVoxelizePass.cpp, and the block in smoke_voxelize.frag. Both
// stages declare the whole struct so their offsets agree.
layout(push_constant) uniform PC
{
    mat4 model;
    vec4 boundsMin;
    vec4 boundsInvSize;
    uvec4 gridSize;
    uvec4 viewportSize;
    uint primitiveIdOffset;
    uint paletteOffset;
    uint skinEnabled;
    uint padding;
} pc;

// Grid-space position of this vertex, in cells: xy index the rasterized slice and z is the depth
// the fragment shader fills from. Passed flat through the rasterizer so the interpolated z is the
// true surface depth at each covered cell centre.
layout(location = 0) out vec3 outGridPos;

void main()
{
    vec3 position = inPosition;
    if (pc.skinEnabled != 0)
    {
        position = (calculateSkinMatrix(gl_VertexIndex, pc.paletteOffset) * vec4(position, 1.0)).xyz;
    }

    vec3 worldPos = (pc.model * vec4(position, 1.0)).xyz;

    // Normalize the box to [0,1], then scale to cells. This is the whole "camera": an orthographic
    // projection down the grid's +Z of the simulation box, so no view matrix is needed.
    vec3 unitPos = (worldPos - pc.boundsMin.xyz) * pc.boundsInvSize.xyz;
    vec3 gridPos = unitPos * vec3(pc.gridSize.xyz);
    outGridPos = gridPos;

    // The viewport covers the whole allocated volume while the grid is only its origin-corner
    // sub-region, so map cells to pixels directly rather than stretching the box across the
    // viewport. That keeps one pixel equal to one cell at any grid size, which is what lets the
    // viewport stay fixed while the grid resizes. The surplus columns fall outside the clip volume
    // and rasterize nothing.
    vec2 pixel = gridPos.xy / vec2(pc.viewportSize.xy);

    // Depth is resolved per-fragment by the XOR, so clip z only has to stay inside the clip volume:
    // geometry in front of and behind the box must still reach the rasterizer, because its surfaces
    // bound the spans that parity fills. Anything outside the box in xy is correctly clipped away,
    // as those columns hold no cells.
    gl_Position = vec4(pixel * 2.0 - 1.0, 0.5, 1.0);
}
