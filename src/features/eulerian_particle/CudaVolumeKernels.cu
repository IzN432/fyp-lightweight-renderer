#include "CudaVolumeKernels.cuh"

#include <cuda_runtime.h>

namespace lr::eulerian
{
namespace
{

// The solver is collocated: every field samples at cell centres, so a cell's index maps directly
// to a texel in each volume and no staggered offsets are needed. Boundary handling clamps reads
// to the grid, which keeps the Laplacian and divergence stencils branch-light in the interior.

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

__device__ inline float4 sampleVector(cudaSurfaceObject_t surface, float3 p, GridInfo grid)
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

    const float4 c000 = readVector(surface, x0, y0, z0, grid);
    const float4 c100 = readVector(surface, x0 + 1, y0, z0, grid);
    const float4 c010 = readVector(surface, x0, y0 + 1, z0, grid);
    const float4 c110 = readVector(surface, x0 + 1, y0 + 1, z0, grid);
    const float4 c001 = readVector(surface, x0, y0, z0 + 1, grid);
    const float4 c101 = readVector(surface, x0 + 1, y0, z0 + 1, grid);
    const float4 c011 = readVector(surface, x0, y0 + 1, z0 + 1, grid);
    const float4 c111 = readVector(surface, x0 + 1, y0 + 1, z0 + 1, grid);

    float4 result;
    result.x = (((c000.x + (c100.x - c000.x) * tx) * (1.0f - ty) + (c010.x + (c110.x - c010.x) * tx) * ty)) *
                   (1.0f - tz) +
               (((c001.x + (c101.x - c001.x) * tx) * (1.0f - ty) + (c011.x + (c111.x - c011.x) * tx) * ty)) * tz;
    result.y = (((c000.y + (c100.y - c000.y) * tx) * (1.0f - ty) + (c010.y + (c110.y - c010.y) * tx) * ty)) *
                   (1.0f - tz) +
               (((c001.y + (c101.y - c001.y) * tx) * (1.0f - ty) + (c011.y + (c111.y - c011.y) * tx) * ty)) * tz;
    result.z = (((c000.z + (c100.z - c000.z) * tx) * (1.0f - ty) + (c010.z + (c110.z - c010.z) * tx) * ty)) *
                   (1.0f - tz) +
               (((c001.z + (c101.z - c001.z) * tx) * (1.0f - ty) + (c011.z + (c111.z - c011.z) * tx) * ty)) * tz;
    result.w = 0.0f;
    return result;
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
// outside of the box — an immovable no-slip wall — so the handling mirrors applyBoundaries: fluid
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
                                           GridInfo grid)
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
                if (isSolid(obstacle, sx, sy, sz, grid))
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
                                               float3 p, GridInfo grid, float ambient)
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
                if (obstacle != 0 && isSolid(obstacle, sx, sy, sz, grid))
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
                              SmokeSourceParams source, float dt)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    // An emitter buried in geometry injects nothing rather than pressurizing the inside of a mesh,
    // which the projection would then have to fight every step.
    if (isSolid(obstacle, static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), grid))
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

    float4 v = readVector(velocity, x, y, z, grid);
    v.x += source.velocity[0] * amount;
    v.y += source.velocity[1] * amount;
    v.z += source.velocity[2] * amount;
    writeVector(velocity, x, y, z, v);
}

// -------------------------------------------------------------------------------------------
// 2. Velocity advection (semi-Lagrangian backtrace)
// -------------------------------------------------------------------------------------------

__global__ void advectVelocity(cudaSurfaceObject_t velocity, cudaSurfaceObject_t velocityNext,
                               cudaSurfaceObject_t obstacle, cudaExtent extent, float dt)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    // Solid cells are held at rest. Zeroing rather than leaving them is what makes the field a
    // genuine no-slip wall: the trilinear taps of neighbouring fluid cells read these cells, so a
    // stale velocity here would keep leaking momentum out of the obstacle forever.
    if (isSolid(obstacle, static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), grid))
    {
        writeVector(velocityNext, x, y, z, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        return;
    }

    const float4   v    = readVector(velocity, x, y, z, grid);
    const float3   back = make_float3((static_cast<float>(x) + 0.5f) - v.x * dt,
                                      (static_cast<float>(y) + 0.5f) - v.y * dt,
                                      (static_cast<float>(z) + 0.5f) - v.z * dt);
    writeVector(velocityNext, x, y, z, sampleVector(velocity, back, grid));
}

// -------------------------------------------------------------------------------------------
// 3. External forces / buoyancy
// -------------------------------------------------------------------------------------------

// Vorticity omega = curl(u), central differences in cell units. Fedkiw et al. equation 9.
__device__ inline float3 curlAt(cudaSurfaceObject_t velocity, int x, int y, int z, GridInfo grid)
{
    const float4 xp = readVector(velocity, x + 1, y, z, grid);
    const float4 xm = readVector(velocity, x - 1, y, z, grid);
    const float4 yp = readVector(velocity, x, y + 1, z, grid);
    const float4 ym = readVector(velocity, x, y - 1, z, grid);
    const float4 zp = readVector(velocity, x, y, z + 1, grid);
    const float4 zm = readVector(velocity, x, y, z - 1, grid);

    return make_float3(0.5f * ((yp.z - ym.z) - (zp.y - zm.y)),
                       0.5f * ((zp.x - zm.x) - (xp.z - xm.z)),
                       0.5f * ((xp.y - xm.y) - (yp.x - ym.x)));
}

// |omega| at one cell, for the eta = grad|omega| difference below.
//
// It has to be the full three-component norm, and the same one at every tap. Differencing the
// absolute value of a single curl component instead — a different component per axis, as this
// once did — is not the gradient of any one scalar field, so the eta it produces points somewhere
// that is not "towards denser vorticity" and the confinement force bears no particular relation to
// equation 10. It still looks like turbulence, which is why it survives an eyeball test.
__device__ inline float curlMagnitude(cudaSurfaceObject_t velocity, int x, int y, int z, GridInfo grid)
{
    const float3 c = curlAt(velocity, x, y, z, grid);
    return sqrtf(c.x * c.x + c.y * c.y + c.z * c.z);
}

__global__ void applyForces(cudaSurfaceObject_t velocity, cudaSurfaceObject_t density,
                            cudaSurfaceObject_t temperature, cudaSurfaceObject_t obstacle, cudaExtent extent,
                            SmokeStepParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    // Buoyancy inside an obstacle would accelerate a wall. Advection already holds these cells at
    // zero, so leaving them untouched is enough.
    if (isSolid(obstacle, static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), grid))
        return;

    const float    d    = readScalar(density, x, y, z, grid);
    const float    t    = readScalar(temperature, x, y, z, grid);
    float4         v    = readVector(velocity, x, y, z, grid);

    // Fedkiw et al. equation 8: hot gas rises, smoke falls under its own weight. The two terms
    // oppose each other, so a parcel climbs only while its heat beats its mass and settles once
    // advectTemperature has cooled it back towards ambient.
    v.y += (params.buoyancy * (t - params.ambientTemperature) - params.gravity * d) * params.dt;

    if (params.vorticity > 0.0f)
    {
        // Vorticity confinement, Fedkiw et al. equations 9-11: push velocity along N x omega to
        // restore the small eddies semi-Lagrangian advection smooths away.
        const int ix = static_cast<int>(x);
        const int iy = static_cast<int>(y);
        const int iz = static_cast<int>(z);

        const float3 curl = curlAt(velocity, ix, iy, iz, grid);

        // eta = grad|omega| (equation 10), central-differenced from the same six neighbours. Each
        // tap is the full |omega| at that cell, so this really is the gradient of one scalar field.
        float3 eta = make_float3(0.5f * (curlMagnitude(velocity, ix + 1, iy, iz, grid) -
                                         curlMagnitude(velocity, ix - 1, iy, iz, grid)),
                                 0.5f * (curlMagnitude(velocity, ix, iy + 1, iz, grid) -
                                         curlMagnitude(velocity, ix, iy - 1, iz, grid)),
                                 0.5f * (curlMagnitude(velocity, ix, iy, iz + 1, grid) -
                                         curlMagnitude(velocity, ix, iy, iz - 1, grid)));

        const float norm = sqrtf(eta.x * eta.x + eta.y * eta.y + eta.z * eta.z);
        if (norm > 1e-8f)
        {
            // N = eta / |eta|, pointing from thinner towards denser vorticity.
            eta.x /= norm;
            eta.y /= norm;
            eta.z /= norm;
            // f_conf = epsilon * h * (N x omega). The grid spacing h is 1 in the cell units the
            // solver works in, so it folds into epsilon; see SmokeStepParams::vorticity for what
            // that costs in resolution independence.
            const float scale = params.vorticity * params.dt;
            v.x += scale * (eta.y * curl.z - eta.z * curl.y);
            v.y += scale * (eta.z * curl.x - eta.x * curl.z);
            v.z += scale * (eta.x * curl.y - eta.y * curl.x);
        }
    }

    if (params.velocityDamp > 0.0f)
    {
        const float keep = fmaxf(0.0f, 1.0f - params.velocityDamp * params.dt);
        v.x *= keep;
        v.y *= keep;
        v.z *= keep;
    }

    writeVector(velocity, x, y, z, v);
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

    // Solid walls: no flow through the border or through an obstacle, so neither has a divergence
    // to correct. Writing zero rather than skipping matters: the Jacobi sweep reads this cell as
    // the right-hand side for its own relaxation.
    //
    // The open face is not such a wall. Flow does cross it, so its row carries a real divergence
    // like any interior cell, and zeroing it here would tell the solve that no mass leaves — the
    // displaced fluid would then have to find a free face elsewhere, which is the sideways push an
    // open top is supposed to remove.
    const bool openFace = onOpenFace(iy, grid, openTop);
    if ((onBorder(ix, iy, iz, grid) && !openFace) || isSolid(obstacle, ix, iy, iz, grid, openTop))
    {
        writeScalar(divergence, ix, iy, iz, 0.0f);
        return;
    }

    const float4 self = readVector(velocity, ix, iy, iz, grid);
    float4       xp   = readVector(velocity, ix + 1, iy, iz, grid);
    float4       xm   = readVector(velocity, ix - 1, iy, iz, grid);
    float4       yp   = readVector(velocity, ix, iy + 1, iz, grid);
    float4       ym   = readVector(velocity, ix, iy - 1, iz, grid);
    float4       zp   = readVector(velocity, ix, iy, iz + 1, grid);
    float4       zm   = readVector(velocity, ix, iy, iz - 1, grid);

    // Where a neighbour is solid, substitute this cell's own velocity for it on that axis. The
    // central difference across the pair then contributes nothing on that side, which is the
    // free-slip no-flux condition (u.n = 0) at the obstacle face: the solve sees no fluid arriving
    // from the wall and no room to push into it, so it builds exactly the pressure that deflects
    // the flow along the surface. Substituting zero instead would read as fluid rushing *out* of
    // the wall and produce a spurious source at every boundary cell.
    if (isSolid(obstacle, ix + 1, iy, iz, grid, openTop)) xp.x = self.x;
    if (isSolid(obstacle, ix - 1, iy, iz, grid, openTop)) xm.x = self.x;
    if (isSolid(obstacle, ix, iy + 1, iz, grid, openTop)) yp.y = self.y;
    if (isSolid(obstacle, ix, iy - 1, iz, grid, openTop)) ym.y = self.y;
    if (isSolid(obstacle, ix, iy, iz + 1, grid, openTop)) zp.z = self.z;
    if (isSolid(obstacle, ix, iy, iz - 1, grid, openTop)) zm.z = self.z;

    const float d = 0.5f * ((xp.x - xm.x) + (yp.y - ym.y) + (zp.z - zm.z));
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

    // The open face is held at ambient, which is the Dirichlet half of the outflow condition and
    // what makes the system non-singular when it is enabled. Writing it every sweep rather than
    // once keeps it from drifting as the swaps alternate the buffers.
    if (onOpenFace(iy, grid, openTop))
    {
        writeScalar(pressureNext, ix, iy, iz, 0.0f);
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

    // An obstacle cell is a wall, not fluid: hold it at rest instead of projecting it.
    if (isSolid(obstacle, ix, iy, iz, grid, openTop))
    {
        writeVector(velocity, ix, iy, iz, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        return;
    }

    // The open face is held at ambient by the solve rather than relaxed, so there is no converged
    // pressure here whose gradient is meaningful. Projecting it anyway would subtract a one-sided
    // gradient the sweeps never balanced; applyBoundaries copies this row's velocity outward from
    // the interior immediately afterwards, so the only thing that projection could do is disagree
    // with the field the vent is supposed to carry.
    if (onOpenFace(iy, grid, openTop))
        return;

    const float centre = readScalar(pressure, ix, iy, iz, grid);

    float4 v = readVector(velocity, ix, iy, iz, grid);
    // The same reflected neighbours the solve used. Using the raw pressure here instead would
    // subtract a gradient the solve never balanced, and the mismatch shows up as flow seeping into
    // the obstacle.
    const float gradX = 0.5f * (neighbourPressure(pressure, obstacle, ix + 1, iy, iz, centre, grid, openTop) -
                                neighbourPressure(pressure, obstacle, ix - 1, iy, iz, centre, grid, openTop));
    const float gradY = 0.5f * (neighbourPressure(pressure, obstacle, ix, iy + 1, iz, centre, grid, openTop) -
                                neighbourPressure(pressure, obstacle, ix, iy - 1, iz, centre, grid, openTop));
    const float gradZ = 0.5f * (neighbourPressure(pressure, obstacle, ix, iy, iz + 1, centre, grid, openTop) -
                                neighbourPressure(pressure, obstacle, ix, iy, iz - 1, centre, grid, openTop));
    v.x -= gradX;
    v.y -= gradY;
    v.z -= gradZ;

    // The projection is only approximately divergence-free after a finite number of Jacobi sweeps,
    // so a little flow into the wall survives it. Clamping the normal component to outward-or-zero
    // on each solid face removes exactly that residue, and is what stops density creeping through a
    // thin obstacle over many steps. Only obstacle faces are clamped here; the box's own border is
    // left to applyBoundaries, which already reflects it.
    if (isSolid(obstacle, ix + 1, iy, iz, grid, openTop)) v.x = fminf(v.x, 0.0f);
    if (isSolid(obstacle, ix - 1, iy, iz, grid, openTop)) v.x = fmaxf(v.x, 0.0f);
    if (isSolid(obstacle, ix, iy + 1, iz, grid, openTop)) v.y = fminf(v.y, 0.0f);
    if (isSolid(obstacle, ix, iy - 1, iz, grid, openTop)) v.y = fmaxf(v.y, 0.0f);
    if (isSolid(obstacle, ix, iy, iz + 1, grid, openTop)) v.z = fminf(v.z, 0.0f);
    if (isSolid(obstacle, ix, iy, iz - 1, grid, openTop)) v.z = fmaxf(v.z, 0.0f);

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
    if (isSolid(obstacle, ix, iy, iz, grid))
    {
        writeScalar(densityNext, ix, iy, iz, 0.0f);
        return;
    }

    const float4   v    = readVector(velocity, x, y, z, grid);
    const float3   back = make_float3((static_cast<float>(x) + 0.5f) - v.x * params.dt,
                                      (static_cast<float>(y) + 0.5f) - v.y * params.dt,
                                      (static_cast<float>(z) + 0.5f) - v.z * params.dt);

    float value = sampleDensityFluid(density, obstacle, back, grid);
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
    if (isSolid(obstacle, ix, iy, iz, grid))
    {
        writeScalar(temperatureNext, ix, iy, iz, params.ambientTemperature);
        return;
    }

    const float4 v    = readVector(velocity, ix, iy, iz, grid);
    const float3 back = make_float3((static_cast<float>(x) + 0.5f) - v.x * params.dt,
                                    (static_cast<float>(y) + 0.5f) - v.y * params.dt,
                                    (static_cast<float>(z) + 0.5f) - v.z * params.dt);

    // Backtrace about the ambient offset rather than the raw value, so the solid taps this drops
    // extrapolate to "ambient air continues" instead of to zero. Same argument as
    // sampleDensityFluid, with a non-zero base: a wall must not cool the gas beside it.
    const float relative = sampleTemperatureFluid(temperature, obstacle, back, grid, params.ambientTemperature);

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

    if (!onBorder(ix, iy, iz, grid))
        return;

    // The inward neighbour this border cell mirrors. Corner cells step inward on every axis they
    // are on the border of.
    const int nx = ix == 0 ? 1 : (ix == grid.width - 1 ? grid.width - 2 : ix);
    const int ny = iy == 0 ? 1 : (iy == grid.height - 1 ? grid.height - 2 : iy);
    const int nz = iz == 0 ? 1 : (iz == grid.depth - 1 ? grid.depth - 2 : iz);

    // No-slip walls: the normal component reflects so the interpolated wall velocity is zero,
    // while the tangential components copy inward.
    float4 v = readVector(velocity, nx, ny, nz, grid);
    if (ix != nx)
        v.x = -v.x;
    // The open face is the exception: copying vy outward instead of negating it lets the flow
    // leave, where the reflection is exactly what turned a rising plume back down into the box.
    // A cell on the top row and a side wall at once still reflects on the side's axis, which is
    // why this only overrides the y term.
    if (iy != ny && !onOpenFace(iy, grid, openTop))
        v.y = -v.y;
    if (iz != nz)
        v.z = -v.z;
    writeVector(velocity, ix, iy, iz, v);

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
        writeScalar(pressure, ix, iy, iz, 0.0f);
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
                               float dt, cudaStream_t stream)
{
    injectSources<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.density, fields.temperature,
                                                          fields.obstacle, extent, source, dt);
    return cudaGetLastError();
}

cudaError_t launchAdvectVelocity(const SmokeFields &fields, cudaExtent extent, float dt, cudaStream_t stream)
{
    advectVelocity<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.velocityNext, fields.obstacle,
                                                           extent, dt);
    return cudaGetLastError();
}

cudaError_t launchApplyForces(const SmokeFields &fields, cudaExtent extent, const SmokeStepParams &params,
                              cudaStream_t stream)
{
    applyForces<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.density, fields.temperature,
                                                        fields.obstacle, extent, params);
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
