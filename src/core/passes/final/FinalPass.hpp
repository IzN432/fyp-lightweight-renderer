#pragma once

#include "core/framegraph/FrameGraph.hpp"

#include <glm/vec4.hpp>
#include <vulkan/vulkan.h>

namespace lr
{

class FinalPass
{
public:
    struct Config
    {
        std::string cameraBufferResourceName;
        VkFormat    swapchainFormat;
        glm::vec4   backgroundColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        bool        showEnvironmentBackground = true;
    };

    explicit FinalPass(Config cfg);

    // Takes effect from the next frame without rebuilding the graph.
    void setBackground(glm::vec4 color, bool showEnvironment)
    {
        m_cfg.backgroundColor          = color;
        m_cfg.showEnvironmentBackground = showEnvironment;
    }

    void build(FrameGraph &fg) const;

private:
    Config m_cfg;
};

} // namespace lr
