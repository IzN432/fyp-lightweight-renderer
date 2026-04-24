#pragma once

#include "core/framegraph/FrameGraph.hpp"

#include <vulkan/vulkan.h>

#include <filesystem>

namespace lr
{

class FinalPass
{
public:
    struct Config
    {
        std::filesystem::path shaderDir;
        VkFormat              swapchainFormat;
    };

    explicit FinalPass(Config cfg);

    void build(FrameGraph &fg) const;

private:
    Config m_cfg;
};

}  // namespace lr
