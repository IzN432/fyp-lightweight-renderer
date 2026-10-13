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

    // Opens the top face of the box, so smoke and the flow carrying it leave instead of being
    // turned back into the domain.
    //
    // With every face closed the pressure problem is pure Neumann, which is a sealed box: the only
    // way out for fluid a rising plume displaces is back down, so the plume drives a recirculation
    // cell and smoke deflected sideways by an obstacle is pushed under rather than around it. That
    // is correct for a sealed box and wrong for the open air a plume is usually meant to read as.
    //
    // Open replaces the top face's conditions with the outflow pair: a Dirichlet p = 0 there,
    // which lets the pressure solve vent the displaced fluid instead of balancing it against a
    // lid, and a velocity border that copies the interior outward rather than reflecting its
    // normal component. The other five faces stay closed walls.
    bool  openTop       = false;
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

    // Obstacle mask: uint32, non-zero where a cell is inside scene geometry. Rasterized by
    // SmokeVoxelizePass on the Vulkan side, read-only here. Zero disables collisions entirely, so
    // a solver with no scene to voxelize runs exactly as it did before the mask existed.
    cudaSurfaceObject_t obstacle     = 0;  // uint32
};

cudaError_t launchInjectSources(const SmokeFields &fields, cudaExtent extent, const SmokeSourceParams &source,
                               float dt, cudaStream_t stream);

cudaError_t launchAdvectVelocity(const SmokeFields &fields, cudaExtent extent, float dt, cudaStream_t stream);

cudaError_t launchApplyForces(const SmokeFields &fields, cudaExtent extent, const SmokeStepParams &params,
                              cudaStream_t stream);

cudaError_t launchComputeDivergence(const SmokeFields &fields, cudaExtent extent, cudaStream_t stream);

// One Jacobi relaxation sweep: reads `pressure`, writes `pressureNext`. The caller swaps between
// sweeps, so an even iteration count leaves the result back in `pressure`.
//
// Takes `openTop` because an open face is a Dirichlet condition on this solve, not something a
// later stage can patch up: the pressure that vents the plume has to be the pressure the sweeps
// converged on.
cudaError_t launchPressureJacobi(const SmokeFields &fields, cudaExtent extent, bool openTop, cudaStream_t stream);

// `openTop` must match what launchPressureJacobi was given: the projection subtracts the gradient
// of the field the solve produced, so the two have to agree about the top face's condition.
cudaError_t launchProjectVelocity(const SmokeFields &fields, cudaExtent extent, bool openTop, cudaStream_t stream);

cudaError_t launchAdvectDensity(const SmokeFields &fields, cudaExtent extent, const SmokeStepParams &params,
                               cudaStream_t stream);

// Clamps the one-cell border: velocity reflects to no-slip and pressure/density mirror inward.
// With `openTop` the top face instead copies the interior outward, so flow and smoke leave there.
cudaError_t launchApplyBoundaries(const SmokeFields &fields, cudaExtent extent, bool openTop, cudaStream_t stream);

cudaError_t launchClearVolume(cudaSurfaceObject_t surface, cudaExtent extent, int channels, cudaStream_t stream);

// Writes an axis-aligned box of solid cells into a uint32 obstacle mask, and clears every cell
// outside it. For tests and debugging: in the application the mask is rasterized from real scene
// geometry by SmokeVoxelizePass, and this would overwrite that. `min` is inclusive and `max`
// exclusive, both in cells, so an empty or inverted range clears the whole volume.
cudaError_t launchFillObstacleBox(cudaSurfaceObject_t obstacle, cudaExtent extent, int3 min, int3 max,
                                  cudaStream_t stream);

cudaError_t launchCopyVolume(cudaSurfaceObject_t destination, cudaSurfaceObject_t source, cudaExtent extent,
                             int channels, cudaStream_t stream);

} // namespace lr::eulerian
