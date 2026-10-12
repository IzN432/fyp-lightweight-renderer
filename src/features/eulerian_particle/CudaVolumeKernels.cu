#include "CudaVolumeKernels.cuh"

#include <cuda_runtime.h>

namespace lr::eulerian
{
namespace
{

__global__ void densityTestPattern(cudaSurfaceObject_t surface, cudaExtent extent, float timeSeconds)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= extent.width || y >= extent.height || z >= extent.depth)
        return;

    const float3 p = make_float3((static_cast<float>(x) + 0.5f) / static_cast<float>(extent.width) - 0.5f,
                                 (static_cast<float>(y) + 0.5f) / static_cast<float>(extent.height) - 0.5f,
                                 (static_cast<float>(z) + 0.5f) / static_cast<float>(extent.depth) - 0.5f);
    const float wobble  = 0.08f * sinf(timeSeconds);
    const float radius2 = (p.x - wobble) * (p.x - wobble) + p.y * p.y + p.z * p.z;
    const float density = expf(-radius2 * 45.0f);
    surf3Dwrite(density, surface, x * sizeof(float), y, z);
}

} // namespace

cudaError_t launchDensityTestPattern(cudaSurfaceObject_t surface, cudaExtent extent, float timeSeconds,
                                     cudaStream_t stream)
{
    constexpr dim3 block{8, 8, 4};
    const dim3 grid{(static_cast<uint32_t>(extent.width) + block.x - 1) / block.x,
                    (static_cast<uint32_t>(extent.height) + block.y - 1) / block.y,
                    (static_cast<uint32_t>(extent.depth) + block.z - 1) / block.z};
    densityTestPattern<<<grid, block, 0, stream>>>(surface, extent, timeSeconds);
    return cudaGetLastError();
}

} // namespace lr::eulerian
