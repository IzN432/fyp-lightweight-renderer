#pragma once

#include "CudaInteropSemaphore.hpp"
#include "CudaVolume.hpp"
#include "CudaVolumeKernels.cuh"

#include <vulkan/vulkan.h>

#include <memory>
#include <string>

namespace lr
{
class ResourceRegistry;
class VulkanContext;
}

namespace lr::eulerian
{

// A grid-based (Eulerian) smoke simulation running on CUDA over Vulkan-exported volumes.
//
// Each step runs the standard incompressible-flow pipeline:
//   1. inject sources        2. advect velocity      3. apply forces / buoyancy
//   4. compute divergence    5. Jacobi pressure solve 6. project velocity
//   7. advect density and temperature                 8. enforce boundaries
//
// The discretization follows Fedkiw, Stam and Jensen, "Visual Simulation of Smoke" (SIGGRAPH
// 2001): scalar fields are cell-centred and velocity is staggered on a MAC grid. Its three face
// components are packed into one float4 volume to preserve the renderer-facing resource. The
// solver uses semi-Lagrangian advection and a Jacobi-relaxed Poisson pressure solve. Buoyancy and
// vorticity confinement follow the same paper's equations 8 to 11: temperature is advected as its own
// field and drives the plume upward while density weighs it down, so a parcel climbs while it is
// hot and spreads once it has cooled. One deliberate departure remains:
//
//   - No viscous diffusion stage, since smoke is modelled as inviscid. SmokeStepParams::
//     velocityDamp is ad-hoc drag, not a solve of the diffusion equation.
//
// Density and velocity live in ResourceRegistry volumes so the renderer can sample them; pressure
// and divergence are registered too, which keeps them inspectable for debug views. Register the
// volumes with registerVolumes() before constructing the solver, or let the constructor do it.
//
// Obstacles are optional and come from outside: Config::obstacleMaskName names a volume that
// SmokeVoxelizePass rasterizes from live scene geometry on the Vulkan side, and stages 2-8 treat
// its set cells as immovable free-slip walls, exactly as they treat the box's own border. The mask
// joins the semaphore handshake below as another named image, so each frame's rasterization is
// ordered before the kernels that read it without any extra synchronization.
//
// Simulation and rendering are ordered by imported external semaphores, not by host waits, so
// step() returns while the kernels are still running. The VulkanContext must therefore enable
// VK_KHR_external_memory and VK_KHR_external_semaphore plus the platform handle extensions
// (..._win32 on Windows, ..._fd on Linux); construction throws if the semaphore ones are missing.
class SmokeSolver
{
public:
    struct Config
    {
        // Resource names for the four registered volumes. The ping-pong copies append "_next".
        std::string       namePrefix        = "smoke";

        // The allocated volume size, fixed for the solver's lifetime. Every kernel bounds-checks
        // against the extent it is handed rather than the texture's, so the simulated region may
        // be smaller than this; see setActiveExtent(). Allocating once at the maximum is what lets
        // the grid be resized without re-importing the volumes or rebuilding the semaphores.
        VkExtent3D        extent            = {64, 64, 64};

        // The simulated sub-region, anchored at the origin corner. Zero means "all of extent".
        VkExtent3D        activeExtent      = {0, 0, 0};

        // An already-registered R32_UINT volume holding the obstacle mask: non-zero cells are
        // treated as immovable free-slip walls, exactly as the box's own border is. The solver does
        // not own, register or write it — SmokeVoxelizePass rasterizes it on the Vulkan side each
        // frame, and the solver only imports it and reads it. It must be exportable and allocated
        // at `extent`, since the kernels index it with the same cell coordinates as the fields.
        //
        // Empty means no collisions at all, which is the standalone demo's case: every kernel then
        // behaves exactly as it did before obstacles existed.
        std::string       obstacleMaskName;

        int               jacobiIterations  = 140;
        SmokeStepParams   step{};
        SmokeSourceParams source{};
    };

    // Registers every volume the solver needs on `resources`, then imports them into CUDA. The
    // registry and context must outlive the solver.
    SmokeSolver(ResourceRegistry &resources, const VulkanContext &context, Config config);
    ~SmokeSolver();

    SmokeSolver(const SmokeSolver &)            = delete;
    SmokeSolver &operator=(const SmokeSolver &) = delete;
    SmokeSolver(SmokeSolver &&) noexcept;
    SmokeSolver &operator=(SmokeSolver &&) noexcept;

    // Registers the volumes a solver with this config will import, without creating one. Useful
    // when the frame graph needs the resources declared before the CUDA device is chosen.
    static void registerVolumes(ResourceRegistry &resources, const Config &config);

    // Enqueues one simulation step and returns without blocking the host.
    //
    // The step is ordered against Vulkan by a pair of imported external semaphores rather than by
    // host waits: a Vulkan submission transitions the volumes and signals, CUDA's stream waits on
    // that signal before the first kernel, and CUDA signals back after the last one. A Vulkan
    // submission consumes that second signal, which is what makes the new contents safe to sample.
    //
    // That consuming submission is issued by endStep(), so a caller that wants the result in
    // Vulkan calls step() then endStep(). The two are separate so a frame loop can put its own
    // work between them and overlap simulation with rendering. Calling step() twice without an
    // intervening endStep() throws rather than deadlocking on the binary semaphores.
    void step();

    // Runs one step with an explicit timestep, overriding Config::step.dt for that step only.
    void step(float dt);

    // Closes the handshake the last step() opened: submits the Vulkan-side wait that orders CUDA's
    // writes before any later Vulkan work, and leaves the volumes safe to sample. Does not block
    // the host. A no-op if no step is outstanding, so it is safe to call defensively.
    void endStep();

    // Blocks until the outstanding step's CUDA work has finished, closing the handshake first.
    // Only for host readback and tests; a frame loop uses endStep() and lets the GPU stay ahead.
    void waitForStep();

    // Whether a step() is outstanding, i.e. CUDA has been handed the volumes and endStep() has not
    // yet taken them back.
    bool stepPending() const;

    // Zeroes every field. Called by the constructor; call again to restart the simulation. Closes
    // any outstanding step and blocks until the clears are done, so it is safe to sample after.
    void reset();

    // The simulated sub-region. Must be at least 3 cells per axis and no larger than the allocated
    // extent. Changing it does not clear the fields: the caller resets, because density left over
    // from a different grid size is meaningless rather than merely stale.
    void       setActiveExtent(VkExtent3D extent);
    VkExtent3D activeExtent() const;

    SmokeStepParams   &stepParams() { return m_config.step; }
    SmokeSourceParams &sourceParams() { return m_config.source; }
    Config            &config() { return m_config; }
    const Config      &config() const { return m_config; }

    // The imported volumes, for rendering hookups and for tests that read fields back. Each
    // reference stays valid for the solver's lifetime.
    // The imported obstacle mask, or null when Config::obstacleMaskName was empty. The solver only
    // reads it; it is exposed so a test or a debug tool can populate it directly (see
    // launchFillObstacleBox) instead of standing up the whole Vulkan voxelizer.
    const CudaVolume *obstacleVolume() const;

    const CudaVolume &densityVolume() const;
    const CudaVolume &velocityVolume() const;
    const CudaVolume &pressureVolume() const;
    const CudaVolume &divergenceVolume() const;
    const CudaVolume &temperatureVolume() const;

    // Resource names of the volumes a renderer samples.
    std::string densityResourceName() const;
    std::string velocityResourceName() const;
    std::string pressureResourceName() const;
    std::string divergenceResourceName() const;
    std::string temperatureResourceName() const;

private:
    struct Impl;

    Config                m_config;
    std::unique_ptr<Impl> m_impl;
};

} // namespace lr::eulerian
