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

__global__ void injectSources(cudaSurfaceObject_t velocity, cudaSurfaceObject_t density, cudaExtent extent,
                              SmokeSourceParams source, float dt)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
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

    float4 v = readVector(velocity, x, y, z, grid);
    v.x += source.velocity[0] * amount;
    v.y += source.velocity[1] * amount;
    v.z += source.velocity[2] * amount;
    writeVector(velocity, x, y, z, v);
}

// -------------------------------------------------------------------------------------------
// 2. Velocity advection (semi-Lagrangian backtrace)
// -------------------------------------------------------------------------------------------

__global__ void advectVelocity(cudaSurfaceObject_t velocity, cudaSurfaceObject_t velocityNext, cudaExtent extent,
                               float dt)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    const float4   v    = readVector(velocity, x, y, z, grid);
    const float3   back = make_float3((static_cast<float>(x) + 0.5f) - v.x * dt,
                                      (static_cast<float>(y) + 0.5f) - v.y * dt,
                                      (static_cast<float>(z) + 0.5f) - v.z * dt);
    writeVector(velocityNext, x, y, z, sampleVector(velocity, back, grid));
}

// -------------------------------------------------------------------------------------------
// 3. External forces / buoyancy
// -------------------------------------------------------------------------------------------

__global__ void applyForces(cudaSurfaceObject_t velocity, cudaSurfaceObject_t density, cudaExtent extent,
                            SmokeStepParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    const float    d    = readScalar(density, x, y, z, grid);
    float4         v    = readVector(velocity, x, y, z, grid);

    // Smoke rises where it is denser than the ambient medium and sinks under its own weight.
    v.y += (params.buoyancy * (d - params.ambient) - params.gravity * d) * params.dt;

    if (params.vorticity > 0.0f)
    {
        // Vorticity confinement: push velocity back along the curl's gradient to restore the small
        // eddies that semi-Lagrangian advection smooths away.
        const float4 xp = readVector(velocity, x + 1, y, z, grid);
        const float4 xm = readVector(velocity, x - 1, y, z, grid);
        const float4 yp = readVector(velocity, x, y + 1, z, grid);
        const float4 ym = readVector(velocity, x, y - 1, z, grid);
        const float4 zp = readVector(velocity, x, y, z + 1, grid);
        const float4 zm = readVector(velocity, x, y, z - 1, grid);

        const float3 curl = make_float3(0.5f * ((yp.z - ym.z) - (zp.y - zm.y)),
                                        0.5f * ((zp.x - zm.x) - (xp.z - xm.z)),
                                        0.5f * ((xp.y - xm.y) - (yp.x - ym.x)));

        // Gradient of |curl|, built from the same six neighbours' curl magnitudes.
        const float mxp = fabsf(0.5f * ((readVector(velocity, x + 1, y + 1, z, grid).z -
                                         readVector(velocity, x + 1, y - 1, z, grid).z) -
                                        (readVector(velocity, x + 1, y, z + 1, grid).y -
                                         readVector(velocity, x + 1, y, z - 1, grid).y)));
        const float mxm = fabsf(0.5f * ((readVector(velocity, x - 1, y + 1, z, grid).z -
                                         readVector(velocity, x - 1, y - 1, z, grid).z) -
                                        (readVector(velocity, x - 1, y, z + 1, grid).y -
                                         readVector(velocity, x - 1, y, z - 1, grid).y)));
        const float myp = fabsf(0.5f * ((readVector(velocity, x, y + 1, z + 1, grid).x -
                                         readVector(velocity, x, y + 1, z - 1, grid).x) -
                                        (readVector(velocity, x + 1, y + 1, z, grid).z -
                                         readVector(velocity, x - 1, y + 1, z, grid).z)));
        const float mym = fabsf(0.5f * ((readVector(velocity, x, y - 1, z + 1, grid).x -
                                         readVector(velocity, x, y - 1, z - 1, grid).x) -
                                        (readVector(velocity, x + 1, y - 1, z, grid).z -
                                         readVector(velocity, x - 1, y - 1, z, grid).z)));
        const float mzp = fabsf(0.5f * ((readVector(velocity, x + 1, y, z + 1, grid).y -
                                         readVector(velocity, x - 1, y, z + 1, grid).y) -
                                        (readVector(velocity, x, y + 1, z + 1, grid).x -
                                         readVector(velocity, x, y - 1, z + 1, grid).x)));
        const float mzm = fabsf(0.5f * ((readVector(velocity, x + 1, y, z - 1, grid).y -
                                         readVector(velocity, x - 1, y, z - 1, grid).y) -
                                        (readVector(velocity, x, y + 1, z - 1, grid).x -
                                         readVector(velocity, x, y - 1, z - 1, grid).x)));

        float3      eta  = make_float3(0.5f * (mxp - mxm), 0.5f * (myp - mym), 0.5f * (mzp - mzm));
        const float norm = sqrtf(eta.x * eta.x + eta.y * eta.y + eta.z * eta.z);
        if (norm > 1e-8f)
        {
            eta.x /= norm;
            eta.y /= norm;
            eta.z /= norm;
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

__global__ void computeDivergence(cudaSurfaceObject_t velocity, cudaSurfaceObject_t divergence, cudaExtent extent)
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

    // Solid walls: no flow through the border, so its divergence contribution is zero.
    if (onBorder(ix, iy, iz, grid))
    {
        writeScalar(divergence, ix, iy, iz, 0.0f);
        return;
    }

    const float4 xp = readVector(velocity, ix + 1, iy, iz, grid);
    const float4 xm = readVector(velocity, ix - 1, iy, iz, grid);
    const float4 yp = readVector(velocity, ix, iy + 1, iz, grid);
    const float4 ym = readVector(velocity, ix, iy - 1, iz, grid);
    const float4 zp = readVector(velocity, ix, iy, iz + 1, grid);
    const float4 zm = readVector(velocity, ix, iy, iz - 1, grid);

    const float d = 0.5f * ((xp.x - xm.x) + (yp.y - ym.y) + (zp.z - zm.z));
    writeScalar(divergence, ix, iy, iz, d);
}

// -------------------------------------------------------------------------------------------
// 5. Jacobi pressure solve
// -------------------------------------------------------------------------------------------

__global__ void pressureJacobi(cudaSurfaceObject_t pressure, cudaSurfaceObject_t pressureNext,
                               cudaSurfaceObject_t divergence, cudaExtent extent)
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

    // Solving laplacian(p) = divergence. clampIndex gives each border cell its own value back as
    // the missing neighbour, which is the pure-Neumann (dp/dn = 0) condition a closed box wants.
    const float sum = readScalar(pressure, ix + 1, iy, iz, grid) + readScalar(pressure, ix - 1, iy, iz, grid) +
                      readScalar(pressure, ix, iy + 1, iz, grid) + readScalar(pressure, ix, iy - 1, iz, grid) +
                      readScalar(pressure, ix, iy, iz + 1, grid) + readScalar(pressure, ix, iy, iz - 1, grid);
    const float d = readScalar(divergence, ix, iy, iz, grid);
    writeScalar(pressureNext, ix, iy, iz, (sum - d) / 6.0f);
}

// -------------------------------------------------------------------------------------------
// 6. Pressure projection
// -------------------------------------------------------------------------------------------

__global__ void projectVelocity(cudaSurfaceObject_t velocity, cudaSurfaceObject_t pressure, cudaExtent extent)
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

    float4      v     = readVector(velocity, ix, iy, iz, grid);
    const float gradX = 0.5f * (readScalar(pressure, ix + 1, iy, iz, grid) - readScalar(pressure, ix - 1, iy, iz, grid));
    const float gradY = 0.5f * (readScalar(pressure, ix, iy + 1, iz, grid) - readScalar(pressure, ix, iy - 1, iz, grid));
    const float gradZ = 0.5f * (readScalar(pressure, ix, iy, iz + 1, grid) - readScalar(pressure, ix, iy, iz - 1, grid));
    v.x -= gradX;
    v.y -= gradY;
    v.z -= gradZ;
    writeVector(velocity, ix, iy, iz, v);
}

// -------------------------------------------------------------------------------------------
// 7. Density advection
// -------------------------------------------------------------------------------------------

__global__ void advectDensity(cudaSurfaceObject_t velocity, cudaSurfaceObject_t density,
                              cudaSurfaceObject_t densityNext, cudaExtent extent, SmokeStepParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (outside(x, y, z, extent))
        return;

    const GridInfo grid = gridInfo(extent);
    const float4   v    = readVector(velocity, x, y, z, grid);
    const float3   back = make_float3((static_cast<float>(x) + 0.5f) - v.x * params.dt,
                                      (static_cast<float>(y) + 0.5f) - v.y * params.dt,
                                      (static_cast<float>(z) + 0.5f) - v.z * params.dt);

    float value = sampleScalar(density, back, grid);
    if (params.densityDamp > 0.0f)
        value *= fmaxf(0.0f, 1.0f - params.densityDamp * params.dt);
    writeScalar(densityNext, x, y, z, fmaxf(0.0f, value));
}

// -------------------------------------------------------------------------------------------
// 8. Boundary conditions
// -------------------------------------------------------------------------------------------

__global__ void applyBoundaries(cudaSurfaceObject_t velocity, cudaSurfaceObject_t density,
                                cudaSurfaceObject_t pressure, cudaExtent extent)
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
    if (iy != ny)
        v.y = -v.y;
    if (iz != nz)
        v.z = -v.z;
    writeVector(velocity, ix, iy, iz, v);

    // Pressure and density mirror inward (zero normal derivative), matching the Neumann condition
    // the Jacobi sweep assumes.
    writeScalar(pressure, ix, iy, iz, readScalar(pressure, nx, ny, nz, grid));
    writeScalar(density, ix, iy, iz, readScalar(density, nx, ny, nz, grid));
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
    injectSources<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.density, extent, source, dt);
    return cudaGetLastError();
}

cudaError_t launchAdvectVelocity(const SmokeFields &fields, cudaExtent extent, float dt, cudaStream_t stream)
{
    advectVelocity<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.velocityNext, extent, dt);
    return cudaGetLastError();
}

cudaError_t launchApplyForces(const SmokeFields &fields, cudaExtent extent, const SmokeStepParams &params,
                              cudaStream_t stream)
{
    applyForces<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.density, extent, params);
    return cudaGetLastError();
}

cudaError_t launchComputeDivergence(const SmokeFields &fields, cudaExtent extent, cudaStream_t stream)
{
    computeDivergence<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.divergence, extent);
    return cudaGetLastError();
}

cudaError_t launchPressureJacobi(const SmokeFields &fields, cudaExtent extent, cudaStream_t stream)
{
    pressureJacobi<<<gridFor(extent), kBlock, 0, stream>>>(fields.pressure, fields.pressureNext, fields.divergence,
                                                           extent);
    return cudaGetLastError();
}

cudaError_t launchProjectVelocity(const SmokeFields &fields, cudaExtent extent, cudaStream_t stream)
{
    projectVelocity<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.pressure, extent);
    return cudaGetLastError();
}

cudaError_t launchAdvectDensity(const SmokeFields &fields, cudaExtent extent, const SmokeStepParams &params,
                               cudaStream_t stream)
{
    advectDensity<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.density, fields.densityNext, extent,
                                                          params);
    return cudaGetLastError();
}

cudaError_t launchApplyBoundaries(const SmokeFields &fields, cudaExtent extent, cudaStream_t stream)
{
    applyBoundaries<<<gridFor(extent), kBlock, 0, stream>>>(fields.velocity, fields.density, fields.pressure, extent);
    return cudaGetLastError();
}

cudaError_t launchClearVolume(cudaSurfaceObject_t surface, cudaExtent extent, int channels, cudaStream_t stream)
{
    clearVolume<<<gridFor(extent), kBlock, 0, stream>>>(surface, extent, channels);
    return cudaGetLastError();
}

cudaError_t launchCopyVolume(cudaSurfaceObject_t destination, cudaSurfaceObject_t source, cudaExtent extent,
                             int channels, cudaStream_t stream)
{
    copyVolume<<<gridFor(extent), kBlock, 0, stream>>>(destination, source, extent, channels);
    return cudaGetLastError();
}

} // namespace lr::eulerian
