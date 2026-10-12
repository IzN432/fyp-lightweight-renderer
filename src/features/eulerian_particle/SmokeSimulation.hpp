#pragma once

#include "features/eulerian_particle/SmokeRenderPass.hpp"
#include "features/eulerian_particle/SmokeSolver.hpp"

#include <glm/vec3.hpp>
#include <vulkan/vulkan.h>

#include <string>

namespace lr
{
class ResourceRegistry;
class VulkanContext;
class FrameGraph;
}

namespace lr::eulerian
{

// Application-facing smoke controller. It owns the CUDA solver, the ray-march pass and every knob
// the two expose, so the app only has to build it, step it once a frame and draw its window.
//
// The render bounds are derived from the grid extent rather than authored: the smoke shader maps
// the box to the volume with a plain normalized divide, so a box whose aspect disagrees with the
// grid's renders the plume stretched. Deriving them keeps the two in step if the grid is resized.
class SmokeSimulation
{
public:
    struct Config
    {
        SmokeSolver::Config solver{};

        // World-space extent of the grid's widest axis. The other axes scale with their cell
        // counts, so cells stay cubic in world space.
        float worldWidth = 2.0f;

        // Render-pass hookups. The defaults match the editor's frame graph.
        std::string cameraBufferResourceName;
        std::string depthResourceName = "gbufferDepth";
        std::string outputImage       = "swapchain";
        VkFormat    outputFormat      = VK_FORMAT_UNDEFINED;

        glm::vec3 color{0.82f, 0.86f, 0.92f};
        float     densityScale = 5.0f;
        int       sampleCount  = 128;
        bool      enabled      = true;

        // Horizontal wander applied to the emitter, in normalized grid units. Keeps the plume from
        // settling into a perfectly straight column. Zero holds the source still.
        float sourceDrift = 0.0f;
    };

    // Registers and imports the solver's volumes; the registry and context must outlive this.
    SmokeSimulation(Config config, ResourceRegistry &resources, const VulkanContext &context);

    void build(FrameGraph &frameGraph) const;

    // Steps the solver and closes the Vulkan handshake, so the density volume is safe to sample by
    // the time this returns. A no-op while disabled or paused.
    void update(float frameDeltaTime);

    // Zeroes every field and restarts the plume.
    void reset();

    // Draws controls only; the caller owns the containing ImGui window.
    void onGUI();

    bool enabled() const { return m_renderPass.config().enabled; }
    bool paused() const { return m_paused; }
    void setPaused(bool paused) { m_paused = paused; }

    const glm::vec3 &boundsMin() const { return m_renderPass.config().boundsMin; }
    const glm::vec3 &boundsMax() const { return m_renderPass.config().boundsMax; }

private:
    static glm::vec3 worldSize(const Config &config);
    void             applySourceDrift();

    Config          m_config;
    SmokeSolver     m_solver;
    SmokeRenderPass m_renderPass;
    bool            m_paused              = false;
    bool            m_singleStepRequested = false;
    float           m_elapsed             = 0.0f;
};

} // namespace lr::eulerian
