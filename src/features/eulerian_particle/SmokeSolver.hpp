#pragma once

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
//   7. advect density        8. enforce boundaries
//
// Density and velocity live in ResourceRegistry volumes so the renderer can sample them; pressure
// and divergence are registered too, which keeps them inspectable for debug views. Register the
// volumes with registerVolumes() before constructing the solver, or let the constructor do it.
class SmokeSolver
{
public:
    struct Config
    {
        // Resource names for the four registered volumes. The ping-pong copies append "_next".
        std::string       namePrefix        = "smoke";
        VkExtent3D        extent            = {64, 64, 64};
        int               jacobiIterations  = 40;
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

    // Runs one simulation step and blocks until CUDA is done, so Vulkan may sample the result.
    void step();

    // Runs one step with an explicit timestep, overriding Config::step.dt for that step only.
    void step(float dt);

    // Zeroes every field. Called by the constructor; call again to restart the simulation.
    void reset();

    SmokeStepParams   &stepParams() { return m_config.step; }
    SmokeSourceParams &sourceParams() { return m_config.source; }
    const Config      &config() const { return m_config; }

    // The imported volumes, for rendering hookups and for tests that read fields back. Each
    // reference stays valid for the solver's lifetime.
    const CudaVolume &densityVolume() const;
    const CudaVolume &velocityVolume() const;
    const CudaVolume &pressureVolume() const;
    const CudaVolume &divergenceVolume() const;

    // Resource names of the volumes a renderer samples.
    std::string densityResourceName() const;
    std::string velocityResourceName() const;
    std::string pressureResourceName() const;
    std::string divergenceResourceName() const;

private:
    struct Impl;

    Config                m_config;
    std::unique_ptr<Impl> m_impl;
};

} // namespace lr::eulerian
