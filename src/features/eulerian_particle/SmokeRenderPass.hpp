#pragma once

#include "core/framegraph/FrameGraph.hpp"

#include <glm/vec3.hpp>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

namespace lr::eulerian
{

// Ray-marches a simulation density volume through an axis-aligned world-space box and alpha
// composites it over the renderer's finished image. Opaque scene depth shortens each ray, so smoke
// behind geometry stays hidden while smoke in front remains visible.
class SmokeRenderPass
{
public:
    struct Config
    {
        std::string densityResourceName;
        std::string cameraBufferResourceName;
        std::string depthResourceName = "gbufferDepth";
        std::string outputImage       = "swapchain";
        VkFormat    outputFormat      = VK_FORMAT_UNDEFINED;

        glm::vec3 boundsMin{-1.0f, 0.0f, -1.0f};
        glm::vec3 boundsMax{1.0f, 2.0f, 1.0f};

        // Fraction of the density volume that is actually simulated, per axis: the box maps to
        // this sub-region of the texture rather than to all of it. One when the simulated grid
        // fills its allocation; smaller after the grid has been resized down.
        glm::vec3 uvScale{1.0f};

        glm::vec3 color{0.82f, 0.86f, 0.92f};
        float     densityScale = 5.0f;
        int       sampleCount  = 128;
        bool      enabled      = true;
    };

    explicit SmokeRenderPass(Config config);

    void build(FrameGraph &frameGraph) const;

    Config       &config() { return m_config; }
    const Config &config() const { return m_config; }

private:
    Config m_config;
    // Incremented by the render lambda each frame to animate the output dither. Mutable because
    // build() installs a const lambda; the frame graph runs it once per frame on one thread.
    mutable uint32_t m_frameIndex = 0;
};

} // namespace lr::eulerian
