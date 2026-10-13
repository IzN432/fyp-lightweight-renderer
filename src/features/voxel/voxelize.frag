#version 450

// Solid voxelization by depth parity.
//
// The vertex shader projects the scene orthographically down the grid's +Z axis, so one fragment
// is one (x,y) grid column and the fragment's own z is where a surface crosses that column. Every
// cell at or beyond that crossing is toggled. A ray entering a closed mesh toggles its remaining
// cells on, the matching exit toggles the cells past it back off, and what survives is exactly the
// span between the two: the interior. Any odd number of crossings leaves a cell set, which is the
// standard point-in-polyhedron parity test evaluated for a whole column at once.
//
// Why parity rather than marking the covered cell: a surface-only mask is a hollow shell, and the
// a volume solver would advect through the empty middle of a closed mesh, while a particle solver
// could tunnel into it. Consumers need interior occupancy, not only cells touched by triangles.
//
// Two consequences worth knowing when reading the mask in RenderDoc:
//   - Order does not matter. XOR commutes, so overlapping draws and the undefined order of
//     fragments in a column both come out the same, which is why this needs no depth test, no
//     sorting and no atomic beyond the XOR itself.
//   - Open meshes (a plane, a mesh with holes) have unpaired crossings, so they smear "solid" from
//     the surface to the far wall of the box. That is inherent to parity; such meshes want the
//     surface-marking variant instead.

layout(set = 0, binding = 11, r32ui) uniform coherent uimage3D obstacleMask;

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

layout(location = 0) in vec3 inGridPos;

void main()
{
    ivec3 grid = ivec3(pc.gridSize.xyz);

    // The rasterizer already restricted us to the box in xy; gl_FragCoord's integer part is the
    // column, taken from the viewport rather than from the interpolated position so it matches the
    // sample the rasterizer actually chose.
    ivec2 column = ivec2(gl_FragCoord.xy);
    if (any(lessThan(column, ivec2(0))) || any(greaterThanEqual(column, grid.xy)))
    {
        return;
    }

    // Round the crossing up to the next cell centre: a surface at z = 4.3 lies past the centre of
    // cell 4, so cell 4 is outside the span and toggling starts at 5. Using the same rule for
    // entry and exit surfaces is what makes the two cancel exactly.
    int first = int(ceil(inGridPos.z - 0.5));
    first = max(first, 0);

    for (int z = first; z < grid.z; ++z)
    {
        // One XOR per cell. imageAtomicXor rather than a load/store pair because several triangles
        // in the same column are in flight at once and parity must survive every one of them.
        imageAtomicXor(obstacleMask, ivec3(column, z), 1u);
    }
}
