#pragma once

#include <cuda_runtime_api.h>

namespace lr::eulerian
{

cudaError_t launchDensityTestPattern(cudaSurfaceObject_t surface, cudaExtent extent, float timeSeconds,
                                     cudaStream_t stream);

// Parameters shared by every stage of one smoke step.
//
// Grid spacing is uniform and implicit: every stage works in cell units, i.e. h = dx = 1. That is
// a consistent non-dimensionalization rather than a missing factor — the 1/dx the textbook
// stencils carry becomes 1, and MAC divergence and pressure gradient use the same face spacing,
// so the pair stays self-consistent. Velocities are therefore in cells per
// second, which is what lets advection backtrace with `dt` alone.
//
// What it costs: the force strengths below are not resolution-independent. Fedkiw's confinement
// force carries an explicit h so that it vanishes as the mesh is refined; with h folded into the
// coefficient, doubling the grid resolution changes how much turbulence a given `vorticity` adds,
// and `buoyancy` shifts the same way. Retune them after a large resolution change rather than
// expecting the look to carry over.
struct SmokeStepParams
{
    float dt            = 1.0f / 60.0f;

    // Buoyancy, Fedkiw et al. equation 8: f = -alpha*rho*z_hat + beta*(T - T_amb)*z_hat.
    //
    // Temperature is what lifts the plume and density is what weighs it down, so the two terms
    // pull against each other rather than both scaling with the smoke. That is what makes a plume
    // stall: it rises while it is hot, cools towards ambientTemperature as it goes, and once
    // beta*(T - T_amb) no longer beats alpha*rho it spreads and falls back instead of climbing
    // forever, which is what buoyancy driven by density alone does.
    //
    // Scale: with the default emitter the gas enters at `temperature` degrees against
    // `ambientTemperature`, so beta*(T - T_amb) at the base is a few hundred times beta, while
    // alpha*rho there is only as large as the injected density. beta therefore belongs around
    // 0.1, not the order-unity value the old per-density force used: 0.1 against a 380 degree
    // source gives roughly twice the lift needed to carry the smoke's own weight, and the rest
    // decays as `cooling` closes the gap to ambient.
    float buoyancy      = 0.1f;   // beta, upward force per degree above ambient
    float gravity       = 1.0f;   // alpha, downward force per unit density
    float ambientTemperature = 20.0f;  // T_amb, the temperature that produces no buoyant force

    // Fraction of the gap to ambientTemperature closed per second, so the plume loses its lift as
    // it rises. Zero holds the injected heat forever, which reverts to a plume that never stalls.
    //
    // The default is tuned against the default emitter and buoyancy: 380 degrees above ambient,
    // losing about two thirds of that per second, which carries a plume up a 48-cell box in a
    // couple of seconds and leaves it spreading rather than still accelerating. Raising it stalls
    // the plume lower, and the buoyancy coefficient has to come up with it to keep the same reach.
    //
    // Both are bounded above by the pressure solve, not by taste: a faster plume needs more Jacobi
    // iterations to stay divergence-free, and past roughly twice these values even the default
    // 140 sweeps visibly stop converging.
    float cooling       = 1.0f;
    // Ad-hoc drag, not viscosity: the solver has no diffusion stage, because smoke is modelled
    // with the inviscid Euler equations. This is a plain per-second decay on the velocity field.
    float velocityDamp  = 0.0f;  // fraction of velocity removed per second
    float densityDamp   = 0.0f;  // fraction of density removed per second (dissipation)
    float vorticity     = 0.0f;  // vorticity-confinement strength (epsilon*h), 0 disables the stage

    // Opens the top face of the box, so smoke and the flow carrying it leave instead of being
    // turned back into the domain.
    //
    // With every face closed the pressure problem is pure Neumann, which is a sealed box: the only
    // way out for fluid a rising plume displaces is back down, so the plume drives a recirculation
    // cell and smoke deflected sideways by an obstacle is pushed under rather than around it. That
    // is correct for a sealed box and wrong for the open air a plume is usually meant to read as.
    //
    // Open replaces the top face's conditions with the outflow pair: a Dirichlet p = 0 outside,
    // which lets the pressure solve vent the displaced fluid instead of balancing it against a
    // lid, and an unconstrained outward MAC face. The other five faces stay closed walls.
    bool  openTop       = false;
};

// A spherical source splatted into the density, temperature and velocity fields each step.
struct SmokeSourceParams
{
    float centre[3] = {0.5f, 0.15f, 0.5f};  // normalized grid position
    float radius    = 0.08f;                // normalized against the grid's largest axis
    float density   = 1.0f;                 // density injected per second
    float velocity[3] = {0.0f, 1.0f, 0.0f}; // velocity injected per second

    // The temperature the emitter drives its cells towards, as an absolute value rather than a
    // rate: injection blends towards it so a cell cannot be heated past the emitter however long
    // it sits there, which an accumulating rate would allow.
    float temperature = 400.0f;
    // Fraction of the gap to `temperature` closed per second inside the emitter.
    float heating     = 8.0f;
};

// Fields the solver reads and writes. Velocity and density are double-buffered because advection
// is not in-place; pressure is double-buffered for the Jacobi sweeps. Divergence is written once
// per step and read by every Jacobi iteration, so it needs no second copy.
struct SmokeFields
{
    // Packed MAC grid: xyz are the velocities on this cell's positive X/Y/Z faces, respectively.
    cudaSurfaceObject_t velocity     = 0;  // float4
    cudaSurfaceObject_t velocityNext = 0;
    cudaSurfaceObject_t density      = 0;  // float
    cudaSurfaceObject_t densityNext  = 0;
    // Temperature, advected alongside density and read by the buoyancy term. Double-buffered for
    // the same reason density is: its advection reads the whole field while writing it.
    cudaSurfaceObject_t temperature     = 0;  // float
    cudaSurfaceObject_t temperatureNext = 0;
    cudaSurfaceObject_t pressure     = 0;  // float
    cudaSurfaceObject_t pressureNext = 0;
    cudaSurfaceObject_t divergence   = 0;  // float

    // Obstacle mask: uint32, non-zero where a cell is inside scene geometry. Rasterized by
    // SmokeVoxelizePass on the Vulkan side, read-only here. Zero disables collisions entirely, so
    // a solver with no scene to voxelize runs exactly as it did before the mask existed.
    cudaSurfaceObject_t obstacle     = 0;  // uint32
};

cudaError_t launchInjectSources(const SmokeFields &fields, cudaExtent extent, const SmokeSourceParams &source,
                               float dt, bool openTop, cudaStream_t stream);

cudaError_t launchAdvectVelocity(const SmokeFields &fields, cudaExtent extent, float dt, bool openTop,
                                 cudaStream_t stream);

cudaError_t launchApplyForces(const SmokeFields &fields, cudaExtent extent, const SmokeStepParams &params,
                              cudaStream_t stream);

// Takes `openTop` for the same reason the solve does: the open face is a cell flow crosses, so it
// carries a divergence, and the right-hand side has to agree with the operator about that.
cudaError_t launchComputeDivergence(const SmokeFields &fields, cudaExtent extent, bool openTop, cudaStream_t stream);

// One Jacobi relaxation sweep: reads `pressure`, writes `pressureNext`. The caller swaps between
// sweeps, so an even iteration count leaves the result back in `pressure`; an odd one leaves it in
// the spare, which the step's copy-back then publishes either way.
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

// Advects temperature along the projected velocity field and relaxes it towards
// SmokeStepParams::ambientTemperature at the cooling rate. Reads `temperature`, writes
// `temperatureNext`, so the caller swaps afterwards exactly as it does for density.
//
// Obstacle cells are held at ambient rather than at zero: they are walls at room temperature, and
// zeroing them would make every obstacle a cold sink that the buoyancy term reads as downward
// force on the smoke beside it.
cudaError_t launchAdvectTemperature(const SmokeFields &fields, cudaExtent extent, const SmokeStepParams &params,
                                    cudaStream_t stream);

// Enforces solid MAC faces and mirrors scalar fields at the five closed domain boundaries. With
// `openTop`, the top face and its advected scalars are left as outflow.
cudaError_t launchApplyBoundaries(const SmokeFields &fields, cudaExtent extent, bool openTop,
                                  float ambientTemperature, cudaStream_t stream);

cudaError_t launchClearVolume(cudaSurfaceObject_t surface, cudaExtent extent, int channels, cudaStream_t stream);

// launchClearVolume for a single-channel field that resets to something other than zero, which is
// what the temperature field needs: an empty simulation is at ambient, not at absolute zero, and
// clearing it to zero would make the whole box read as ice-cold air on the first step.
cudaError_t launchFillScalar(cudaSurfaceObject_t surface, cudaExtent extent, float value, cudaStream_t stream);

// Writes an axis-aligned box of solid cells into a uint32 obstacle mask, and clears every cell
// outside it. For tests and debugging: in the application the mask is rasterized from real scene
// geometry by SmokeVoxelizePass, and this would overwrite that. `min` is inclusive and `max`
// exclusive, both in cells, so an empty or inverted range clears the whole volume.
cudaError_t launchFillObstacleBox(cudaSurfaceObject_t obstacle, cudaExtent extent, int3 min, int3 max,
                                  cudaStream_t stream);

// Test/debug analogue of the scene voxelizer for a curved obstacle. Centre and radius are in cell
// coordinates; cells whose centres lie inside the sphere are marked solid.
cudaError_t launchFillObstacleSphere(cudaSurfaceObject_t obstacle, cudaExtent extent, float3 centre,
                                     float radius, cudaStream_t stream);

cudaError_t launchCopyVolume(cudaSurfaceObject_t destination, cudaSurfaceObject_t source, cudaExtent extent,
                             int channels, cudaStream_t stream);

// Writes the shear field u = (0, exp(-(x - centre)^2 / falloff), 0) into a float4 velocity volume.
//
// For tests of the vorticity stage. Its curl is purely along z and its |omega| has a known shape,
// so the confinement force it produces can be checked against the closed form of Fedkiw equations
// 10-11 rather than against an eyeball judgement of turbulence. Not used by the application.
cudaError_t launchFillShearField(cudaSurfaceObject_t velocity, cudaExtent extent, float centre, float falloff,
                                 cudaStream_t stream);

} // namespace lr::eulerian
