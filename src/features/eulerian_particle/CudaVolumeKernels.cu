#include "CudaVolumeKernels.cuh"

#include <cuda_runtime.h>

namespace lr::eulerian
{
namespace
{

// Scalars sample at cell centres while velocity uses a staggered MAC layout packed into float4;
// see the obstacle helpers below for the exact component positions. One cell still maps to one
// texel in every allocation, but the three velocity components have distinct spatial offsets.

constexpr dim3 kBlock{8, 8, 4};

dim3 gridFor(cudaExtent extent)
{
    return dim3{(static_cast<uint32_t>(extent.width) + kBlock.x - 1) / kBlock.x,
                (static_cast<uint32_t>(extent.height) + kBlock.y - 1) / kBlock.y,
                (static_cast<uint32_t>(extent.depth) + kBlock.z - 1) / kBlock.z};
}

struct GridInfo
{
    int width  = 0;
    int height = 0;
    int depth  = 0;
};

__device__ inline GridInfo gridInfo(cudaExtent extent)
{
    return GridInfo{static_cast<int>(extent.width), static_cast<int>(extent.height), static_cast<int>(extent.depth)};
}

__device__ inline int clampIndex(int value, int limit)
{
    return value < 0 ? 0 : (value >= limit ? limit - 1 : value);
}

// surf3Dwrite/read take a byte offset on x only; these wrappers keep that detail in one place.
__device__ inline float readScalar(cudaSurfaceObject_t surface, int x, int y, int z, GridInfo grid)
{
    const int cx = clampIndex(x, grid.width);
    const int cy = clampIndex(y, grid.height);
    const int cz = clampIndex(z, grid.depth);
    float     value;
    surf3Dread(&value, surface, cx * static_cast<int>(sizeof(float)), cy, cz);
    return value;
}

__device__ inline void writeScalar(cudaSurfaceObject_t surface, int x, int y, int z, float value)
{
    surf3Dwrite(value, surface, x * static_cast<int>(sizeof(float)), y, z);
}

__device__ inline float4 readVector(cudaSurfaceObject_t surface, int x, int y, int z, GridInfo grid)
{
    const int cx = clampIndex(x, grid.width);
    const int cy = clampIndex(y, grid.height);
    const int cz = clampIndex(z, grid.depth);
    float4    value;
    surf3Dread(&value, surface, cx * static_cast<int>(sizeof(float4)), cy, cz);
    return value;
}

__device__ inline void writeVector(cudaSurfaceObject_t surface, int x, int y, int z, float4 value)
{
    surf3Dwrite(value, surface, x * static_cast<int>(sizeof(float4)), y, z);
}

// Trilinear sample at a cell-centre coordinate, clamped at the border. Surfaces cannot be filtered
// by the texture units the way a cudaTextureObject_t can, so the eight-tap blend is explicit.
__device__ inline float sampleScalar(cudaSurfaceObject_t surface, float3 p, GridInfo grid)
{
    const float px = p.x - 0.5f;
    const float py = p.y - 0.5f;
    const float pz = p.z - 0.5f;
    const int   x0 = static_cast<int>(floorf(px));
    const int   y0 = static_cast<int>(floorf(py));
    const int   z0 = static_cast<int>(floorf(pz));
    const float tx = px - static_cast<float>(x0);
    const float ty = py - static_cast<float>(y0);
    const float tz = pz - static_cast<float>(z0);

    const float c000 = readScalar(surface, x0, y0, z0, grid);
    const float c100 = readScalar(surface, x0 + 1, y0, z0, grid);
    const float c010 = readScalar(surface, x0, y0 + 1, z0, grid);
    const float c110 = readScalar(surface, x0 + 1, y0 + 1, z0, grid);
    const float c001 = readScalar(surface, x0, y0, z0 + 1, grid);
    const float c101 = readScalar(surface, x0 + 1, y0, z0 + 1, grid);
    const float c011 = readScalar(surface, x0, y0 + 1, z0 + 1, grid);
    const float c111 = readScalar(surface, x0 + 1, y0 + 1, z0 + 1, grid);

    const float c00 = c000 + (c100 - c000) * tx;
    const float c10 = c010 + (c110 - c010) * tx;
    const float c01 = c001 + (c101 - c001) * tx;
    const float c11 = c011 + (c111 - c011) * tx;
    const float c0  = c00 + (c10 - c00) * ty;
    const float c1  = c01 + (c11 - c01) * ty;
    return c0 + (c1 - c0) * tz;
}

__device__ inline bool outside(uint32_t x, uint32_t y, uint32_t z, cudaExtent extent)
{
    return x >= extent.width || y >= extent.height || z >= extent.depth;
}

__device__ inline bool onBorder(int x, int y, int z, GridInfo grid)
{
    return x == 0 || y == 0 || z == 0 || x == grid.width - 1 || y == grid.height - 1 || z == grid.depth - 1;
}

// The open face, when there is one: the top row of cells, +y. Everything that treats the box as
// sealed asks this first, so the five closed faces keep their existing behaviour exactly.
__device__ inline bool onOpenFace(int y, GridInfo grid, bool openTop)
{
    return openTop && y == grid.height - 1;
}

// -------------------------------------------------------------------------------------------
// Obstacles
//
// The mask is a solid (not shell) occupancy volume rasterized by SmokeVoxelizePass: one uint per
// cell, non-zero inside scene geometry. The solver treats a solid cell exactly as it treats the
// outside of the box — an immovable free-slip wall — so the handling mirrors applyBoundaries: fluid
// cannot flow into it, pressure takes a zero normal derivative across its face, and nothing is
// advected out of it.
//
// A zero surface means "no mask imported", which every helper below reads as fully fluid. That
// keeps the uncollided behaviour bit-identical for the standalone demo, which has no scene.
// -------------------------------------------------------------------------------------------

// `openTop` makes the open face win over the mask on its row. A voxelized scene ceiling lands in
// the top row like any other geometry, and because every kernel tests isSolid before onOpenFace, a
// solid cell there would otherwise turn the vent back into a lid — silently, since the toggle is
// still on. Reporting that row as fluid is what keeps "open top" meaning open regardless of what
// the scene put there. The five sealed faces are unaffected: they are not the open face.
__device__ inline bool isSolid(cudaSurfaceObject_t obstacle, int x, int y, int z, GridInfo grid,
                               bool openTop = false)
{
    if (openTop && y >= grid.height - 1)
        return false;
    if (obstacle == 0)
        return false;
    const int cx = clampIndex(x, grid.width);
    const int cy = clampIndex(y, grid.height);
    const int cz = clampIndex(z, grid.depth);
    uint32_t  value;
    // Read as uint, matching the r32ui the voxelizer's imageAtomicXor writes. Reading it as float
    // would reinterpret the bits and make 1 a denormal that compares false against zero.
    surf3Dread(&value, obstacle, cx * static_cast<int>(sizeof(uint32_t)), cy, cz);
    return value != 0u;
}

// MAC velocity packing. A single float4 volume is retained for Vulkan interop, but its components
// no longer describe one collocated vector. At storage cell (i,j,k):
//   x = u(i+1, j+1/2, k+1/2), y = v(i+1/2, j+1, k+1/2),
//   z = w(i+1/2, j+1/2, k+1).
// The negative box faces are implicit zero-valued faces; the positive faces occupy the final
// storage index on their respective axis. This is the standard staggered MAC arrangement, packed
// only to avoid changing the public velocity resource from one volume into three.
__device__ inline bool blockedUFace(cudaSurfaceObject_t obstacle, int x, int y, int z, GridInfo grid,
                                    bool openTop)
{
    if (x < 0 || x >= grid.width || y < 0 || y >= grid.height || z < 0 || z >= grid.depth)
        return true;
    if (x == grid.width - 1)
        return true;
    return isSolid(obstacle, x, y, z, grid, openTop) ||
           isSolid(obstacle, x + 1, y, z, grid, openTop);
}

__device__ inline bool blockedVFace(cudaSurfaceObject_t obstacle, int x, int y, int z, GridInfo grid,
                                    bool openTop)
{
    if (x < 0 || x >= grid.width || y < 0 || y >= grid.height || z < 0 || z >= grid.depth)
        return true;
    if (y == grid.height - 1)
        return !openTop;
    return isSolid(obstacle, x, y, z, grid, openTop) ||
           isSolid(obstacle, x, y + 1, z, grid, openTop);
}

__device__ inline bool blockedWFace(cudaSurfaceObject_t obstacle, int x, int y, int z, GridInfo grid,
                                    bool openTop)
{
    if (x < 0 || x >= grid.width || y < 0 || y >= grid.height || z < 0 || z >= grid.depth)
        return true;
    if (z == grid.depth - 1)
        return true;
    return isSolid(obstacle, x, y, z, grid, openTop) ||
           isSolid(obstacle, x, y, z + 1, grid, openTop);
}

__device__ inline float storedVelocityComponent(cudaSurfaceObject_t velocity, int x, int y, int z,
                                                int component, GridInfo grid)
{
    if (x < 0 || x >= grid.width || y < 0 || y >= grid.height || z < 0 || z >= grid.depth)
        return 0.0f;
    const float4 value = readVector(velocity, x, y, z, grid);
    return component == 0 ? value.x : (component == 1 ? value.y : value.z);
}

__device__ inline bool blockedStoredFace(cudaSurfaceObject_t obstacle, int x, int y, int z,
                                         int component, GridInfo grid, bool openTop)
{
    if (component == 0) return blockedUFace(obstacle, x, y, z, grid, openTop);
    if (component == 1) return blockedVFace(obstacle, x, y, z, grid, openTop);
    return blockedWFace(obstacle, x, y, z, grid, openTop);
}

// Interpolate one staggered component at a world-in-grid position. Solid face taps are omitted and
// the remaining weights renormalized. In particular, tangential flow beside a solid is not blended
// with the zero velocity inside it, which was the artificial boundary viscosity in the collocated
// implementation.
__device__ inline float sampleMacComponent(cudaSurfaceObject_t velocity, cudaSurfaceObject_t obstacle,
                                           float3 p, int component, GridInfo grid, bool openTop)
{
    const float3 offset = component == 0 ? make_float3(1.0f, 0.5f, 0.5f)
                         : component == 1 ? make_float3(0.5f, 1.0f, 0.5f)
                                          : make_float3(0.5f, 0.5f, 1.0f);
    const float qx = p.x - offset.x;
    const float qy = p.y - offset.y;
    const float qz = p.z - offset.z;
    const int   x0 = static_cast<int>(floorf(qx));
    const int   y0 = static_cast<int>(floorf(qy));
    const int   z0 = static_cast<int>(floorf(qz));
    const float tx = qx - static_cast<float>(x0);
    const float ty = qy - static_cast<float>(y0);
    const float tz = qz - static_cast<float>(z0);

    float sum = 0.0f;
    float weight = 0.0f;
    for (int k = 0; k < 2; ++k)
    {
        const float wz = k == 0 ? 1.0f - tz : tz;
        for (int j = 0; j < 2; ++j)
        {
            const float wy = j == 0 ? 1.0f - ty : ty;
            for (int i = 0; i < 2; ++i)
            {
                const int sx = x0 + i;
                const int sy = y0 + j;
                const int sz = z0 + k;
                if (blockedStoredFace(obstacle, sx, sy, sz, component, grid, openTop))
                    continue;
                const float wx = i == 0 ? 1.0f - tx : tx;
                const float w = wx * wy * wz;
                sum += w * storedVelocityComponent(velocity, sx, sy, sz, component, grid);
                weight += w;
            }
        }
    }
    return weight > 1e-4f ? sum / weight : 0.0f;
}

__device__ inline float3 sampleMacVelocity(cudaSurfaceObject_t velocity, cudaSurfaceObject_t obstacle,
                                           float3 p, GridInfo grid, bool openTop)
{
    return make_float3(sampleMacComponent(velocity, obstacle, p, 0, grid, openTop),
                       sampleMacComponent(velocity, obstacle, p, 1, grid, openTop),
                       sampleMacComponent(velocity, obstacle, p, 2, grid, openTop));
}

__device__ inline bool pointInSolid(cudaSurfaceObject_t obstacle, float3 p, GridInfo grid, bool openTop)
{
    const int x = static_cast<int>(floorf(p.x));
    const int y = static_cast<int>(floorf(p.y));
    const int z = static_cast<int>(floorf(p.z));
    if (x < 0 || x >= grid.width || z < 0 || z >= grid.depth || y < 0)
        return true;
    if (y >= grid.height)
        return !openTop;
    return isSolid(obstacle, x, y, z, grid, openTop);
}

// Fedkiw et al. clip a semi-Lagrangian characteristic when it enters an occupied voxel. Marching
// at half-cell intervals is sufficient for this binary mask, then a short bisection places the
// sample immediately on the fluid side of the first crossed face.
__device__ inline float3 clipBacktrace(cudaSurfaceObject_t obstacle, float3 start, float3 end,
                                       GridInfo grid, bool openTop)
{
    const float3 delta = make_float3(end.x - start.x, end.y - start.y, end.z - start.z);
    const float length = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    const int steps = max(1, static_cast<int>(ceilf(length * 2.0f)));
    float3 previous = start;
    for (int step = 1; step <= steps; ++step)
    {
        const float t = static_cast<float>(step) / static_cast<float>(steps);
        float3 current = make_float3(start.x + delta.x * t, start.y + delta.y * t, start.z + delta.z * t);
        if (pointInSolid(obstacle, current, grid, openTop))
        {
            float3 fluid = previous;
            float3 solid = current;
            for (int iteration = 0; iteration < 6; ++iteration)
            {
                const float3 mid = make_float3(0.5f * (fluid.x + solid.x), 0.5f * (fluid.y + solid.y),
                                               0.5f * (fluid.z + solid.z));
                if (pointInSolid(obstacle, mid, grid, openTop)) solid = mid;
                else fluid = mid;
            }
            return fluid;
        }
        previous = current;
    }
    return end;
}

__device__ inline float sourceAmountAt(float3 p, SmokeSourceParams source, GridInfo grid,
                                       float longest, float radius, float dt)
{
    const float dx = (p.x - source.centre[0] * static_cast<float>(grid.width)) / longest;
    const float dy = (p.y - source.centre[1] * static_cast<float>(grid.height)) / longest;
    const float dz = (p.z - source.centre[2] * static_cast<float>(grid.depth)) / longest;
    return expf(-(dx * dx + dy * dy + dz * dz) / (radius * radius)) * dt;
}

// sampleScalar for density, with the solid taps left out of the blend instead of contributing the
// zero that advectDensity writes into obstacle cells.
//
// This is about mass, not about where the plume goes. The plain eight-tap blend treats an obstacle
// cell as a cell that genuinely holds no smoke, so a backtrace landing beside a wall reads a
// density diluted toward zero by however many of its taps are solid — up to four of eight against
// a flat face. Every step then loses a fraction of the smoke in the layer touching the obstacle,
// which is dissipation the solver never asked for and that densityDamp cannot account for.
//
// Dropping the solid taps and renormalizing by the surviving weight is a zero-order extrapolation
// of density out of the obstacle: a wall now reads as "the smoke beside it continues", which is
// what the one-sided difference on a staggered grid would see. With every tap solid there is no
// fluid nearby to extrapolate from, so the result is zero.
//
// It does not make a deflected plume climb back over an obstacle. In a box whose every boundary is
// no-flux, the smoke pushed aside has nowhere to go but down — the plume drives a recirculation
// cell, and the channels beside an obstacle carry a downdraft even with no obstacle in the grid at
// all. That is the domain, not this blend.
__device__ inline float sampleDensityFluid(cudaSurfaceObject_t density, cudaSurfaceObject_t obstacle, float3 p,
                                           GridInfo grid, bool openTop)
{
    if (obstacle == 0)
        return sampleScalar(density, p, grid);

    const float px = p.x - 0.5f;
    const float py = p.y - 0.5f;
    const float pz = p.z - 0.5f;
    const int   x0 = static_cast<int>(floorf(px));
    const int   y0 = static_cast<int>(floorf(py));
    const int   z0 = static_cast<int>(floorf(pz));
    const float tx = px - static_cast<float>(x0);
    const float ty = py - static_cast<float>(y0);
    const float tz = pz - static_cast<float>(z0);

    float sum    = 0.0f;
    float weight = 0.0f;
    for (int k = 0; k < 2; ++k)
    {
        const float wz = k == 0 ? 1.0f - tz : tz;
        for (int j = 0; j < 2; ++j)
        {
            const float wy = j == 0 ? 1.0f - ty : ty;
            for (int i = 0; i < 2; ++i)
            {
                const float wx = i == 0 ? 1.0f - tx : tx;
                const int   sx = x0 + i;
                const int   sy = y0 + j;
                const int   sz = z0 + k;
                if (isSolid(obstacle, sx, sy, sz, grid, openTop))
                    continue;
                const float w = wx * wy * wz;
                sum += w * readScalar(density, sx, sy, sz, grid);
                weight += w;
            }
        }
    }

    // Guard the weight rather than testing it for zero: a backtrace landing almost exactly on the
    // one fluid corner of an otherwise solid neighbourhood leaves a legitimate but tiny weight,
    // and dividing by it would amplify that corner's density enormously.
    return weight > 1e-4f ? sum / weight : 0.0f;
}

// sampleDensityFluid for temperature, returning the value *relative to ambient*.
//
// Temperature's "nothing here" is ambient rather than zero, so the solid-tap exclusion has to
// extrapolate towards ambient too. Differencing each tap against ambient first turns that back
// into the same problem sampleDensityFluid solves — a field whose empty state is zero — and the
// caller adds ambient back. With every tap solid the result is zero, i.e. exactly ambient.
__device__ inline float sampleTemperatureFluid(cudaSurfaceObject_t temperature, cudaSurfaceObject_t obstacle,
                                               float3 p, GridInfo grid, bool openTop, float ambient)
{
    const float px = p.x - 0.5f;
    const float py = p.y - 0.5f;
    const float pz = p.z - 0.5f;
    const int   x0 = static_cast<int>(floorf(px));
    const int   y0 = static_cast<int>(floorf(py));
    const int   z0 = static_cast<int>(floorf(pz));
    const float tx = px - static_cast<float>(x0);
    const float ty = py - static_cast<float>(y0);
    const float tz = pz - static_cast<float>(z0);

    float sum    = 0.0f;
    float weight = 0.0f;
    for (int k = 0; k < 2; ++k)
    {
        const float wz = k == 0 ? 1.0f - tz : tz;
        for (int j = 0; j < 2; ++j)
        {
            const float wy = j == 0 ? 1.0f - ty : ty;
            for (int i = 0; i < 2; ++i)
            {
                const float wx = i == 0 ? 1.0f - tx : tx;
                const int   sx = x0 + i;
                const int   sy = y0 + j;
                const int   sz = z0 + k;
                if (obstacle != 0 && isSolid(obstacle, sx, sy, sz, grid, openTop))
                    continue;
                const float w = wx * wy * wz;
                sum += w * (readScalar(temperature, sx, sy, sz, grid) - ambient);
                weight += w;
            }
        }
    }

    return weight > 1e-4f ? sum / weight : 0.0f;
}

__global__ void densityTestPattern(cudaSurfaceObject_t surface, cudaExtent extent, float timeSeconds)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const float3 p = make_float3((static_cast<float>(x) + 0.5f) / static_cast<float>(extent.width) - 0.5f,
                                 (static_cast<float>(y) + 0.5f) / static_cast<float>(extent.height) - 0.5f,
                                 (static_cast<float>(z) + 0.5f) / static_cast<float>(extent.depth) - 0.5f);
    const float wobble  = 0.08f * sinf(timeSeconds);
    const float radius2 = (p.x - wobble) * (p.x - wobble) + p.y * p.y + p.z * p.z;
    const float density = expf(-radius2 * 45.0f);
    surf3Dwrite(density, surface, x * sizeof(float), y, z);
}

// -------------------------------------------------------------------------------------------
// 1. Source injection
// -------------------------------------------------------------------------------------------

__global__ void injectSources(cudaSurfaceObject_t velocity, cudaSurfaceObject_t density,
                              cudaSurfaceObject_t temperature, cudaSurfaceObject_t obstacle, cudaExtent extent,
                              SmokeSourceParams source, float dt, bool openTop)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    // An emitter buried in geometry injects nothing rather than pressurizing the inside of a mesh,
    // which the projection would then have to fight every step.
    if (isSolid(obstacle, static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), grid, openTop))
        return;

    // Distances are measured in normalized space scaled by the largest axis so a spherical source
    // stays spherical on a non-cube grid.
    const float longest = fmaxf(static_cast<float>(grid.width),
                                fmaxf(static_cast<float>(grid.height), static_cast<float>(grid.depth)));
    const float dx = ((static_cast<float>(x) + 0.5f) - source.centre[0] * static_cast<float>(grid.width)) / longest;
    const float dy = ((static_cast<float>(y) + 0.5f) - source.centre[1] * static_cast<float>(grid.height)) / longest;
    const float dz = ((static_cast<float>(z) + 0.5f) - source.centre[2] * static_cast<float>(grid.depth)) / longest;
    const float distance2 = dx * dx + dy * dy + dz * dz;
    const float radius    = fmaxf(source.radius, 1e-6f);
    // Gaussian falloff rather than a hard sphere: a sharp edge injects high-frequency divergence
    // the pressure solve then has to spend iterations removing.
    const float falloff = expf(-distance2 / (radius * radius));
    if (falloff < 1e-4f)
        return;

    const float amount = falloff * dt;
    writeScalar(density, x, y, z, readScalar(density, x, y, z, grid) + source.density * amount);

    // Temperature blends towards the emitter's value rather than accumulating like density does.
    // Heat is a state the emitter holds its cells at, not a quantity it pours in: an accumulating
    // rate would let a cell sitting in the source climb without bound, and the buoyancy term would
    // turn that into an ever-accelerating jet.
    const float heatBlend = fminf(1.0f, fmaxf(0.0f, source.heating * amount));
    const float current   = readScalar(temperature, x, y, z, grid);
    writeScalar(temperature, x, y, z, current + (source.temperature - current) * heatBlend);

    // Each packed component lives at a different MAC face, so evaluate the source at the three
    // actual face centres rather than splatting a collocated vector at the scalar cell centre.
    float4 v = readVector(velocity, x, y, z, grid);
    if (!blockedUFace(obstacle, static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), grid, openTop))
        v.x += source.velocity[0] * sourceAmountAt(
            make_float3(static_cast<float>(x) + 1.0f, static_cast<float>(y) + 0.5f,
                        static_cast<float>(z) + 0.5f), source, grid, longest, radius, dt);
    else v.x = 0.0f;
    if (!blockedVFace(obstacle, static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), grid, openTop))
        v.y += source.velocity[1] * sourceAmountAt(
            make_float3(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 1.0f,
                        static_cast<float>(z) + 0.5f), source, grid, longest, radius, dt);
    else v.y = 0.0f;
    if (!blockedWFace(obstacle, static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), grid, openTop))
        v.z += source.velocity[2] * sourceAmountAt(
            make_float3(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f,
                        static_cast<float>(z) + 1.0f), source, grid, longest, radius, dt);
    else v.z = 0.0f;
    writeVector(velocity, x, y, z, v);
}

// -------------------------------------------------------------------------------------------
// 2. Velocity advection (semi-Lagrangian backtrace)
// -------------------------------------------------------------------------------------------

__global__ void advectVelocity(cudaSurfaceObject_t velocity, cudaSurfaceObject_t velocityNext,
                               cudaSurfaceObject_t obstacle, cudaExtent extent, float dt, bool openTop)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    const int ix = static_cast<int>(x);
    const int iy = static_cast<int>(y);
    const int iz = static_cast<int>(z);
    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const float3 positions[3] = {
        make_float3(static_cast<float>(x) + 1.0f, static_cast<float>(y) + 0.5f,
                    static_cast<float>(z) + 0.5f),
        make_float3(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 1.0f,
                    static_cast<float>(z) + 0.5f),
        make_float3(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f,
                    static_cast<float>(z) + 1.0f)};

    for (int component = 0; component < 3; ++component)
    {
        if (blockedStoredFace(obstacle, ix, iy, iz, component, grid, openTop))
            continue;
        const float3 at = positions[component];
        const float3 transport = sampleMacVelocity(velocity, obstacle, at, grid, openTop);
        float3 back = make_float3(at.x - transport.x * dt, at.y - transport.y * dt,
                                  at.z - transport.z * dt);
        back = clipBacktrace(obstacle, at, back, grid, openTop);
        const float value = sampleMacComponent(velocity, obstacle, back, component, grid, openTop);
        if (component == 0) result.x = value;
        else if (component == 1) result.y = value;
        else result.z = value;
    }
    writeVector(velocityNext, x, y, z, result);
}

// -------------------------------------------------------------------------------------------
// 3. External forces / buoyancy
// -------------------------------------------------------------------------------------------

__device__ inline float3 macCurlAt(cudaSurfaceObject_t velocity, cudaSurfaceObject_t obstacle,
                                   float3 p, GridInfo grid, bool openTop)
{
    const float3 xp = sampleMacVelocity(velocity, obstacle, make_float3(p.x + 1.0f, p.y, p.z), grid, openTop);
    const float3 xm = sampleMacVelocity(velocity, obstacle, make_float3(p.x - 1.0f, p.y, p.z), grid, openTop);
    const float3 yp = sampleMacVelocity(velocity, obstacle, make_float3(p.x, p.y + 1.0f, p.z), grid, openTop);
    const float3 ym = sampleMacVelocity(velocity, obstacle, make_float3(p.x, p.y - 1.0f, p.z), grid, openTop);
    const float3 zp = sampleMacVelocity(velocity, obstacle, make_float3(p.x, p.y, p.z + 1.0f), grid, openTop);
    const float3 zm = sampleMacVelocity(velocity, obstacle, make_float3(p.x, p.y, p.z - 1.0f), grid, openTop);
    return make_float3(0.5f * ((yp.z - ym.z) - (zp.y - zm.y)),
                       0.5f * ((zp.x - zm.x) - (xp.z - xm.z)),
                       0.5f * ((xp.y - xm.y) - (yp.x - ym.x)));
}

__device__ inline float macCurlMagnitude(cudaSurfaceObject_t velocity, cudaSurfaceObject_t obstacle,
                                         float3 p, GridInfo grid, bool openTop)
{
    const float3 curl = macCurlAt(velocity, obstacle, p, grid, openTop);
    return sqrtf(curl.x * curl.x + curl.y * curl.y + curl.z * curl.z);
}

__device__ inline float3 confinementAt(cudaSurfaceObject_t velocity, cudaSurfaceObject_t obstacle,
                                       float3 p, GridInfo grid, bool openTop, float scale)
{
    const float3 curl = macCurlAt(velocity, obstacle, p, grid, openTop);
    float3 eta = make_float3(
        0.5f * (macCurlMagnitude(velocity, obstacle, make_float3(p.x + 1.0f, p.y, p.z), grid, openTop) -
                macCurlMagnitude(velocity, obstacle, make_float3(p.x - 1.0f, p.y, p.z), grid, openTop)),
        0.5f * (macCurlMagnitude(velocity, obstacle, make_float3(p.x, p.y + 1.0f, p.z), grid, openTop) -
                macCurlMagnitude(velocity, obstacle, make_float3(p.x, p.y - 1.0f, p.z), grid, openTop)),
        0.5f * (macCurlMagnitude(velocity, obstacle, make_float3(p.x, p.y, p.z + 1.0f), grid, openTop) -
                macCurlMagnitude(velocity, obstacle, make_float3(p.x, p.y, p.z - 1.0f), grid, openTop)));
    const float norm = sqrtf(eta.x * eta.x + eta.y * eta.y + eta.z * eta.z);
    if (norm <= 1e-8f)
        return make_float3(0.0f, 0.0f, 0.0f);
    eta.x /= norm;
    eta.y /= norm;
    eta.z /= norm;
    return make_float3(scale * (eta.y * curl.z - eta.z * curl.y),
                       scale * (eta.z * curl.x - eta.x * curl.z),
                       scale * (eta.x * curl.y - eta.y * curl.x));
}

__global__ void applyForces(cudaSurfaceObject_t velocity, cudaSurfaceObject_t velocityNext,
                            cudaSurfaceObject_t density,
                            cudaSurfaceObject_t temperature, cudaSurfaceObject_t obstacle, cudaExtent extent,
                            SmokeStepParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    const int ix = static_cast<int>(x);
    const int iy = static_cast<int>(y);
    const int iz = static_cast<int>(z);
    const float3 positions[3] = {
        make_float3(static_cast<float>(x) + 1.0f, static_cast<float>(y) + 0.5f,
                    static_cast<float>(z) + 0.5f),
        make_float3(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 1.0f,
                    static_cast<float>(z) + 0.5f),
        make_float3(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f,
                    static_cast<float>(z) + 1.0f)};
    float4 v = readVector(velocity, x, y, z, grid);

    // Apply confinement at each component's own face centre. This retains Fedkiw's N x omega force
    // after moving velocity off cell centres.
    if (params.vorticity > 0.0f)
    {
        const float scale = params.vorticity * params.dt;
        if (!blockedUFace(obstacle, ix, iy, iz, grid, params.openTop))
            v.x += confinementAt(velocity, obstacle, positions[0], grid, params.openTop, scale).x;
        if (!blockedVFace(obstacle, ix, iy, iz, grid, params.openTop))
            v.y += confinementAt(velocity, obstacle, positions[1], grid, params.openTop, scale).y;
        if (!blockedWFace(obstacle, ix, iy, iz, grid, params.openTop))
            v.z += confinementAt(velocity, obstacle, positions[2], grid, params.openTop, scale).z;
    }

    // Fedkiw et al. equation 8, evaluated at the v face. Scalars remain cell-centred and are
    // interpolated to that face; the solid-aware samplers avoid treating a wall as cold vacuum.
    if (!blockedVFace(obstacle, ix, iy, iz, grid, params.openTop))
    {
        const float d = sampleDensityFluid(density, obstacle, positions[1], grid, params.openTop);
        const float relativeTemperature = sampleTemperatureFluid(
            temperature, obstacle, positions[1], grid, params.openTop, params.ambientTemperature);
        v.y += (params.buoyancy * relativeTemperature - params.gravity * d) * params.dt;
    }

    if (params.velocityDamp > 0.0f)
    {
        const float keep = fmaxf(0.0f, 1.0f - params.velocityDamp * params.dt);
        v.x *= keep;
        v.y *= keep;
        v.z *= keep;
    }

    if (blockedUFace(obstacle, ix, iy, iz, grid, params.openTop)) v.x = 0.0f;
    if (blockedVFace(obstacle, ix, iy, iz, grid, params.openTop)) v.y = 0.0f;
    if (blockedWFace(obstacle, ix, iy, iz, grid, params.openTop)) v.z = 0.0f;

    writeVector(velocityNext, x, y, z, v);
}

// -------------------------------------------------------------------------------------------
// 4. Divergence
// -------------------------------------------------------------------------------------------

__global__ void computeDivergence(cudaSurfaceObject_t velocity, cudaSurfaceObject_t divergence,
                                  cudaSurfaceObject_t obstacle, cudaExtent extent, bool openTop)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    const int      ix   = static_cast<int>(x);
    const int      iy   = static_cast<int>(y);
    const int      iz   = static_cast<int>(z);

    // Divergence is the net flux through this cell's six MAC faces. Solid and closed-domain faces
    // contribute exactly zero; unlike the old collocated central difference, an upward flow into a
    // sphere is therefore visible to the pressure right-hand side instead of cancelling itself.
    if (isSolid(obstacle, ix, iy, iz, grid, openTop))
    {
        writeScalar(divergence, ix, iy, iz, 0.0f);
        return;
    }

    const float4 self = readVector(velocity, ix, iy, iz, grid);
    const float uRight = blockedUFace(obstacle, ix, iy, iz, grid, openTop) ? 0.0f : self.x;
    const float uLeft = ix == 0 || blockedUFace(obstacle, ix - 1, iy, iz, grid, openTop)
                            ? 0.0f
                            : readVector(velocity, ix - 1, iy, iz, grid).x;
    const float vTop = blockedVFace(obstacle, ix, iy, iz, grid, openTop) ? 0.0f : self.y;
    const float vBottom = iy == 0 || blockedVFace(obstacle, ix, iy - 1, iz, grid, openTop)
                              ? 0.0f
                              : readVector(velocity, ix, iy - 1, iz, grid).y;
    const float wFront = blockedWFace(obstacle, ix, iy, iz, grid, openTop) ? 0.0f : self.z;
    const float wBack = iz == 0 || blockedWFace(obstacle, ix, iy, iz - 1, grid, openTop)
                            ? 0.0f
                            : readVector(velocity, ix, iy, iz - 1, grid).z;

    const float d = (uRight - uLeft) + (vTop - vBottom) + (wFront - wBack);
    writeScalar(divergence, ix, iy, iz, d);
}

// -------------------------------------------------------------------------------------------
// 5. Jacobi pressure solve
// -------------------------------------------------------------------------------------------

// A neighbour's pressure for the Laplacian, with a solid neighbour reflecting the centre cell's own
// value. That is the same dp/dn = 0 condition clampIndex already gives the box's border, applied to
// the interior faces an obstacle introduces, and it keeps the stencil's six-neighbour divisor valid
// without a per-cell count.
__device__ inline float neighbourPressure(cudaSurfaceObject_t pressure, cudaSurfaceObject_t obstacle, int x, int y,
                                          int z, float centre, GridInfo grid, bool openTop)
{
    if (isSolid(obstacle, x, y, z, grid, openTop))
        return centre;
    // Past the open face the pressure is ambient, which is the zero this solve measures against.
    // Returning it rather than letting clampIndex mirror the top row back is the whole Dirichlet
    // condition: the cell below sees somewhere for its pressure to fall away to, so the solve
    // vents the plume instead of balancing it against a lid.
    if (openTop && y >= grid.height)
        return 0.0f;
    return readScalar(pressure, x, y, z, grid);
}

__global__ void pressureJacobi(cudaSurfaceObject_t pressure, cudaSurfaceObject_t pressureNext,
                               cudaSurfaceObject_t divergence, cudaSurfaceObject_t obstacle, cudaExtent extent,
                               bool openTop)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    const int      ix   = static_cast<int>(x);
    const int      iy   = static_cast<int>(y);
    const int      iz   = static_cast<int>(z);

    const float centre = readScalar(pressure, ix, iy, iz, grid);

    // Interior of an obstacle: there is no fluid here to solve for, so the cell just carries its
    // current value forward. Propagating it rather than zeroing keeps the reflected neighbour
    // values above consistent between sweeps, and this cell is never projected anyway.
    if (isSolid(obstacle, ix, iy, iz, grid, openTop))
    {
        writeScalar(pressureNext, ix, iy, iz, centre);
        return;
    }

    // Solving laplacian(p) = divergence. clampIndex gives each border cell its own value back as
    // the missing neighbour, which is the pure-Neumann (dp/dn = 0) condition a closed box wants;
    // neighbourPressure extends the same rule to obstacle faces, and overrides it with ambient
    // past an open one.
    const float sum = neighbourPressure(pressure, obstacle, ix + 1, iy, iz, centre, grid, openTop) +
                      neighbourPressure(pressure, obstacle, ix - 1, iy, iz, centre, grid, openTop) +
                      neighbourPressure(pressure, obstacle, ix, iy + 1, iz, centre, grid, openTop) +
                      neighbourPressure(pressure, obstacle, ix, iy - 1, iz, centre, grid, openTop) +
                      neighbourPressure(pressure, obstacle, ix, iy, iz + 1, centre, grid, openTop) +
                      neighbourPressure(pressure, obstacle, ix, iy, iz - 1, centre, grid, openTop);
    const float d = readScalar(divergence, ix, iy, iz, grid);
    writeScalar(pressureNext, ix, iy, iz, (sum - d) / 6.0f);
}

// -------------------------------------------------------------------------------------------
// 6. Pressure projection
// -------------------------------------------------------------------------------------------

__global__ void projectVelocity(cudaSurfaceObject_t velocity, cudaSurfaceObject_t pressure,
                                cudaSurfaceObject_t obstacle, cudaExtent extent, bool openTop)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    const int      ix   = static_cast<int>(x);
    const int      iy   = static_cast<int>(y);
    const int      iz   = static_cast<int>(z);

    float4 v = readVector(velocity, ix, iy, iz, grid);

    // Each component is projected from the two pressure samples straddling that same MAC face.
    // This makes gradient and divergence exact discrete adjoints and removes the collocated
    // checkerboard mode described by Bridson and Mueller-Fischer.
    if (blockedUFace(obstacle, ix, iy, iz, grid, openTop))
        v.x = 0.0f;
    else
        v.x -= readScalar(pressure, ix + 1, iy, iz, grid) - readScalar(pressure, ix, iy, iz, grid);

    if (blockedVFace(obstacle, ix, iy, iz, grid, openTop))
        v.y = 0.0f;
    else if (iy == grid.height - 1)
        v.y -= 0.0f - readScalar(pressure, ix, iy, iz, grid);
    else
        v.y -= readScalar(pressure, ix, iy + 1, iz, grid) - readScalar(pressure, ix, iy, iz, grid);

    if (blockedWFace(obstacle, ix, iy, iz, grid, openTop))
        v.z = 0.0f;
    else
        v.z -= readScalar(pressure, ix, iy, iz + 1, grid) - readScalar(pressure, ix, iy, iz, grid);

    writeVector(velocity, ix, iy, iz, v);
}

// -------------------------------------------------------------------------------------------
// 7. Density advection
// -------------------------------------------------------------------------------------------

__global__ void advectDensity(cudaSurfaceObject_t velocity, cudaSurfaceObject_t density,
                              cudaSurfaceObject_t densityNext, cudaSurfaceObject_t obstacle, cudaExtent extent,
                              SmokeStepParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    const int      ix   = static_cast<int>(x);
    const int      iy   = static_cast<int>(y);
    const int      iz   = static_cast<int>(z);

    // Obstacles hold no smoke. Zeroing them every step also keeps the trilinear taps of the fluid
    // cells beside them from blending in density that an earlier frame left inside geometry — which
    // is what would happen if an obstacle moved into an already-smoky region.
    if (isSolid(obstacle, ix, iy, iz, grid, params.openTop))
    {
        writeScalar(densityNext, ix, iy, iz, 0.0f);
        return;
    }

    const float3 at = make_float3(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f,
                                  static_cast<float>(z) + 0.5f);
    const float3 v = sampleMacVelocity(velocity, obstacle, at, grid, params.openTop);
    float3 back = make_float3(at.x - v.x * params.dt, at.y - v.y * params.dt,
                              at.z - v.z * params.dt);
    back = clipBacktrace(obstacle, at, back, grid, params.openTop);

    float value = sampleDensityFluid(density, obstacle, back, grid, params.openTop);
    if (params.densityDamp > 0.0f)
        value *= fmaxf(0.0f, 1.0f - params.densityDamp * params.dt);
    writeScalar(densityNext, x, y, z, fmaxf(0.0f, value));
}

// -------------------------------------------------------------------------------------------
// 7b. Temperature advection
// -------------------------------------------------------------------------------------------

__global__ void advectTemperature(cudaSurfaceObject_t velocity, cudaSurfaceObject_t temperature,
                                  cudaSurfaceObject_t temperatureNext, cudaSurfaceObject_t obstacle,
                                  cudaExtent extent, SmokeStepParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    const int      ix   = static_cast<int>(x);
    const int      iy   = static_cast<int>(y);
    const int      iz   = static_cast<int>(z);

    // An obstacle is a wall at room temperature, not a cold void. Holding it at ambient is what
    // keeps it neutral in the buoyancy term: zeroing it instead would make every wall a heat sink
    // that reads as (0 - T_amb) < 0, i.e. a downward force on the smoke touching it.
    if (isSolid(obstacle, ix, iy, iz, grid, params.openTop))
    {
        writeScalar(temperatureNext, ix, iy, iz, params.ambientTemperature);
        return;
    }

    const float3 at = make_float3(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f,
                                  static_cast<float>(z) + 0.5f);
    const float3 v = sampleMacVelocity(velocity, obstacle, at, grid, params.openTop);
    float3 back = make_float3(at.x - v.x * params.dt, at.y - v.y * params.dt,
                              at.z - v.z * params.dt);
    back = clipBacktrace(obstacle, at, back, grid, params.openTop);

    // Backtrace about the ambient offset rather than the raw value, so the solid taps this drops
    // extrapolate to "ambient air continues" instead of to zero. Same argument as
    // sampleDensityFluid, with a non-zero base: a wall must not cool the gas beside it.
    const float relative = sampleTemperatureFluid(temperature, obstacle, back, grid, params.openTop,
                                                  params.ambientTemperature);

    // Newton cooling towards ambient. Applied as a fraction of the remaining gap per second, so it
    // approaches ambient without crossing it however large dt is.
    float value = params.ambientTemperature + relative;
    if (params.cooling > 0.0f)
    {
        const float keep = fmaxf(0.0f, 1.0f - params.cooling * params.dt);
        value            = params.ambientTemperature + relative * keep;
    }
    writeScalar(temperatureNext, x, y, z, value);
}

// -------------------------------------------------------------------------------------------
// 8. Boundary conditions
// -------------------------------------------------------------------------------------------

__global__ void applyBoundaries(cudaSurfaceObject_t velocity, cudaSurfaceObject_t density,
                                cudaSurfaceObject_t temperature, cudaSurfaceObject_t pressure,
                                cudaSurfaceObject_t obstacle, cudaExtent extent, bool openTop,
                                float ambientTemperature)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    const int      ix   = static_cast<int>(x);
    const int      iy   = static_cast<int>(y);
    const int      iz   = static_cast<int>(z);

    // Obstacle interiors are settled last, after the step's copy-backs have landed, so what the
    // renderer samples and what the next step's stencils read agree: at rest and empty. Pressure is
    // left alone — the Jacobi sweep carries it forward, and overwriting it here would discard the
    // value the reflected neighbours depend on.
    if (isSolid(obstacle, ix, iy, iz, grid, openTop))
    {
        writeVector(velocity, ix, iy, iz, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        writeScalar(density, ix, iy, iz, 0.0f);
        writeScalar(temperature, ix, iy, iz, ambientTemperature);
        return;
    }

    // Set boundary-face velocities directly. With a MAC grid there are no ghost cell vectors to
    // reflect: each stored component is already the normal velocity on one concrete face.
    float4 packedVelocity = readVector(velocity, ix, iy, iz, grid);
    if (blockedUFace(obstacle, ix, iy, iz, grid, openTop)) packedVelocity.x = 0.0f;
    if (blockedVFace(obstacle, ix, iy, iz, grid, openTop)) packedVelocity.y = 0.0f;
    if (blockedWFace(obstacle, ix, iy, iz, grid, openTop)) packedVelocity.z = 0.0f;
    writeVector(velocity, ix, iy, iz, packedVelocity);

    if (!onBorder(ix, iy, iz, grid))
        return;

    // The inward neighbour this border cell mirrors. Corner cells step inward on every axis they
    // are on the border of.
    const int nx = ix == 0 ? 1 : (ix == grid.width - 1 ? grid.width - 2 : ix);
    const int ny = iy == 0 ? 1 : (iy == grid.height - 1 ? grid.height - 2 : iy);
    const int nz = iz == 0 ? 1 : (iz == grid.depth - 1 ? grid.depth - 2 : iz);

    // Pressure and density mirror inward (zero normal derivative), matching the Neumann condition
    // the Jacobi sweep assumes.
    //
    // On the open face pressure is held at the same ambient zero the solve used, and density is
    // left alone rather than mirrored: mirroring would copy the interior's smoke back into the
    // border every step, which reads as a lid that the plume piles up against. advectDensity
    // already backtraces this row like any other, so what stays here is what genuinely flowed in,
    // and the next step carries it out through the open face.
    if (onOpenFace(iy, grid, openTop))
    {
        return;
    }

    writeScalar(pressure, ix, iy, iz, readScalar(pressure, nx, ny, nz, grid));
    writeScalar(density, ix, iy, iz, readScalar(density, nx, ny, nz, grid));
    // Temperature mirrors inward like density, which is the adiabatic wall: no heat flows through
    // the box's sides, so the border carries whatever the interior beside it holds.
    writeScalar(temperature, ix, iy, iz, readScalar(temperature, nx, ny, nz, grid));
}

__global__ void clearVolume(cudaSurfaceObject_t surface, cudaExtent extent, int channels)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    if (channels == 4)
        writeVector(surface, x, y, z, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    else
        writeScalar(surface, x, y, z, 0.0f);
}

__global__ void fillObstacleBox(cudaSurfaceObject_t obstacle, cudaExtent extent, int3 boxMin, int3 boxMax)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const int      ix     = static_cast<int>(x);
    const int      iy     = static_cast<int>(y);
    const int      iz     = static_cast<int>(z);
    const bool     inside = ix >= boxMin.x && ix < boxMax.x && iy >= boxMin.y && iy < boxMax.y && iz >= boxMin.z &&
                        iz < boxMax.z;
    // Written as uint to match the r32ui the voxelizer's atomics use, which is also how isSolid
    // reads it back.
    surf3Dwrite(inside ? 1u : 0u, obstacle, ix * static_cast<int>(sizeof(uint32_t)), iy, iz);
}

__global__ void fillObstacleSphere(cudaSurfaceObject_t obstacle, cudaExtent extent, float3 centre, float radius)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const float dx = static_cast<float>(x) + 0.5f - centre.x;
    const float dy = static_cast<float>(y) + 0.5f - centre.y;
    const float dz = static_cast<float>(z) + 0.5f - centre.z;
    const bool inside = dx * dx + dy * dy + dz * dz <= radius * radius;
    surf3Dwrite(inside ? 1u : 0u, obstacle, x * static_cast<int>(sizeof(uint32_t)), y, z);
}

__global__ void copyVolume(cudaSurfaceObject_t destination, cudaSurfaceObject_t source, cudaExtent extent,
                           int channels)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    if (channels == 4)
        writeVector(destination, x, y, z, readVector(source, x, y, z, grid));
    else
        writeScalar(destination, x, y, z, readScalar(source, x, y, z, grid));
}

__global__ void fillScalar(cudaSurfaceObject_t surface, cudaExtent extent, float value)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    writeScalar(surface, x, y, z, value);
}

// u = (0, exp(-(x - centre)^2 / falloff), 0). See launchFillShearField for why.
__global__ void fillShearField(cudaSurfaceObject_t velocity, cudaExtent extent, float centre, float falloff)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const float dx = (static_cast<float>(x) + 0.5f) - centre;
    writeVector(velocity, x, y, z, make_float4(0.0f, expf(-dx * dx / falloff), 0.0f, 0.0f));
}

} // namespace

cudaError_t launchDensityTestPattern(cudaSurfaceObject_t surface, cudaExtent extent, float timeSeconds,
                                     cudaStream_t stream)
{
    densityTestPattern<<<gridFor(extent), kBlock, 0, stream>>>(surface, extent, timeSeconds);
    return cudaGetLastError();
}

cudaError_t launchInjectSources(const SmokeFields &fields, cudaExtent extent, const SmokeSourceParams &source,
                               float dt, bool openTop, cudaStream_t stream)
{
    injectSources<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.density, fields.temperature,
                                                          fields.obstacle, extent, source, dt, openTop);
    return cudaGetLastError();
}

cudaError_t launchAdvectVelocity(const SmokeFields &fields, cudaExtent extent, float dt, bool openTop,
                                 cudaStream_t stream)
{
    advectVelocity<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.velocityNext, fields.obstacle,
                                                           extent, dt, openTop);
    return cudaGetLastError();
}

cudaError_t launchApplyForces(const SmokeFields &fields, cudaExtent extent, const SmokeStepParams &params,
                              cudaStream_t stream)
{
    applyForces<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.velocityNext, fields.density,
                                                        fields.temperature, fields.obstacle, extent, params);
    return cudaGetLastError();
}

cudaError_t launchComputeDivergence(const SmokeFields &fields, cudaExtent extent, bool openTop, cudaStream_t stream)
{
    computeDivergence<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.divergence, fields.obstacle,
                                                              extent, openTop);
    return cudaGetLastError();
}

cudaError_t launchPressureJacobi(const SmokeFields &fields, cudaExtent extent, bool openTop, cudaStream_t stream)
{
    pressureJacobi<<<gridFor(extent), kBlock, 0, stream>>>(fields.pressure, fields.pressureNext, fields.divergence,
                                                           fields.obstacle, extent, openTop);
    return cudaGetLastError();
}

cudaError_t launchProjectVelocity(const SmokeFields &fields, cudaExtent extent, bool openTop, cudaStream_t stream)
{
    projectVelocity<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.pressure, fields.obstacle, extent,
                                                            openTop);
    return cudaGetLastError();
}

cudaError_t launchAdvectDensity(const SmokeFields &fields, cudaExtent extent, const SmokeStepParams &params,
                               cudaStream_t stream)
{
    advectDensity<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.density, fields.densityNext,
                                                          fields.obstacle, extent, params);
    return cudaGetLastError();
}

cudaError_t launchApplyBoundaries(const SmokeFields &fields, cudaExtent extent, bool openTop,
                                  float ambientTemperature, cudaStream_t stream)
{
    applyBoundaries<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.density, fields.temperature,
                                                            fields.pressure, fields.obstacle, extent, openTop,
                                                            ambientTemperature);
    return cudaGetLastError();
}

cudaError_t launchAdvectTemperature(const SmokeFields &fields, cudaExtent extent, const SmokeStepParams &params,
                                    cudaStream_t stream)
{
    advectTemperature<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.temperature,
                                                              fields.temperatureNext, fields.obstacle, extent,
                                                              params);
    return cudaGetLastError();
}

cudaError_t launchFillScalar(cudaSurfaceObject_t surface, cudaExtent extent, float value, cudaStream_t stream)
{
    fillScalar<<<gridFor(extent), kBlock, 0, stream>>>(surface, extent, value);
    return cudaGetLastError();
}

cudaError_t launchClearVolume(cudaSurfaceObject_t surface, cudaExtent extent, int channels, cudaStream_t stream)
{
    clearVolume<<<gridFor(extent), kBlock, 0, stream>>>(surface, extent, channels);
    return cudaGetLastError();
}

cudaError_t launchFillObstacleBox(cudaSurfaceObject_t obstacle, cudaExtent extent, int3 min, int3 max,
                                  cudaStream_t stream)
{
    fillObstacleBox<<<gridFor(extent), kBlock, 0, stream>>>(obstacle, extent, min, max);
    return cudaGetLastError();
}

cudaError_t launchFillObstacleSphere(cudaSurfaceObject_t obstacle, cudaExtent extent, float3 centre,
                                     float radius, cudaStream_t stream)
{
    fillObstacleSphere<<<gridFor(extent), kBlock, 0, stream>>>(obstacle, extent, centre, radius);
    return cudaGetLastError();
}

cudaError_t launchCopyVolume(cudaSurfaceObject_t destination, cudaSurfaceObject_t source, cudaExtent extent,
                             int channels, cudaStream_t stream)
{
    copyVolume<<<gridFor(extent), kBlock, 0, stream>>>(destination, source, extent, channels);
    return cudaGetLastError();
}

cudaError_t launchFillShearField(cudaSurfaceObject_t velocity, cudaExtent extent, float centre, float falloff,
                                 cudaStream_t stream)
{
    fillShearField<<<gridFor(extent), kBlock, 0, stream>>>(velocity, extent, centre, falloff);
    return cudaGetLastError();
}

} // namespace lr::eulerian
