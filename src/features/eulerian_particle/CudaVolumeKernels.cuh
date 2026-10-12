#pragma once

#include <cuda_runtime_api.h>

namespace lr::eulerian
{

cudaError_t launchDensityTestPattern(cudaSurfaceObject_t surface, cudaExtent extent, float timeSeconds,
                                     cudaStream_t stream);

} // namespace lr::eulerian
