#pragma once

#include "core/framegraph/FrameGraph.hpp"

#include <vulkan/vulkan.h>

#include <filesystem>

namespace lr
{

class PbrPass
{
public:
    struct Config
    {
        std::filesystem::path   shaderDir;
        std::string             cameraBufferResourceName;
        std::string             lightBufferResourceName;
        uint32_t                numLights;
        VkFormat                swapchainFormat;
    };

    explicit PbrPass(Config cfg);

    void build(FrameGraph &fg) const;

private:
    Config m_cfg;
};

}  // namespace lr
