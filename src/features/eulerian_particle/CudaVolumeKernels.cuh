#pragma once

#include <cuda_runtime_api.h>

namespace lr::eulerian
{

cudaError_t launchDensityTestPattern(cudaSurfaceObject_t surface, cudaExtent extent, float timeSeconds,
                                     cudaStream_t stream);

// Parameters shared by every stage of one smoke step. Grid spacing is uniform and implicit: the
// solver works in cell units, so `dt` is pre-divided by the cell size where a stage needs it.
struct SmokeStepParams
{
    float dt            = 1.0f / 60.0f;
    float buoyancy      = 1.0f;  // upward force per unit density
    float ambient       = 0.0f;  // density that produces no buoyant force
    float gravity       = 0.0f;  // downward force per unit density, applied with buoyancy
    float velocityDamp  = 0.0f;  // fraction of velocity removed per second
    float densityDamp   = 0.0f;  // fraction of density removed per second (dissipation)
    float vorticity     = 0.0f;  // vorticity-confinement strength, 0 disables the stage
};

// A spherical source splatted into the density and velocity fields each step.
struct SmokeSourceParams
{
    float centre[3] = {0.5f, 0.15f, 0.5f};  // normalized grid position
    float radius    = 0.08f;                // normalized against the grid's largest axis
    float density   = 1.0f;                 // density injected per second
    float velocity[3] = {0.0f, 1.0f, 0.0f}; // velocity injected per second
};

// Fields the solver reads and writes. Velocity and density are double-buffered because advection
// is not in-place; pressure is double-buffered for the Jacobi sweeps. Divergence is written once
// per step and read by every Jacobi iteration, so it needs no second copy.
struct SmokeFields
{
    cudaSurfaceObject_t velocity     = 0;  // float4, xyz = velocity
    cudaSurfaceObject_t velocityNext = 0;
    cudaSurfaceObject_t density      = 0;  // float
    cudaSurfaceObject_t densityNext  = 0;
    cudaSurfaceObject_t pressure     = 0;  // float
    cudaSurfaceObject_t pressureNext = 0;
    cudaSurfaceObject_t divergence   = 0;  // float
};

cudaError_t launchInjectSources(const SmokeFields &fields, cudaExtent extent, const SmokeSourceParams &source,
                               float dt, cudaStream_t stream);

cudaError_t launchAdvectVelocity(const SmokeFields &fields, cudaExtent extent, float dt, cudaStream_t stream);

cudaError_t launchApplyForces(const SmokeFields &fields, cudaExtent extent, const SmokeStepParams &params,
                              cudaStream_t stream);

cudaError_t launchComputeDivergence(const SmokeFields &fields, cudaExtent extent, cudaStream_t stream);

// One Jacobi relaxation sweep: reads `pressure`, writes `pressureNext`. The caller swaps between
// sweeps, so an even iteration count leaves the result back in `pressure`.
cudaError_t launchPressureJacobi(const SmokeFields &fields, cudaExtent extent, cudaStream_t stream);

cudaError_t launchProjectVelocity(const SmokeFields &fields, cudaExtent extent, cudaStream_t stream);

cudaError_t launchAdvectDensity(const SmokeFields &fields, cudaExtent extent, const SmokeStepParams &params,
                               cudaStream_t stream);

// Clamps the one-cell border: velocity reflects to no-slip and pressure/density mirror inward.
cudaError_t launchApplyBoundaries(const SmokeFields &fields, cudaExtent extent, cudaStream_t stream);

cudaError_t launchClearVolume(cudaSurfaceObject_t surface, cudaExtent extent, int channels, cudaStream_t stream);

cudaError_t launchCopyVolume(cudaSurfaceObject_t destination, cudaSurfaceObject_t source, cudaExtent extent,
                             int channels, cudaStream_t stream);

} // namespace lr::eulerian
