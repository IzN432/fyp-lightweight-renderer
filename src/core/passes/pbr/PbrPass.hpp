#pragma once

#include "core/framegraph/FrameGraph.hpp"

#include <vulkan/vulkan.h>

namespace lr
{

class PbrPass
{
public:
    struct Config
    {
        std::string             cameraBufferResourceName;
        std::string             lightBufferResourceName;
        uint32_t                numLights;
        uint32_t                pfMips;
    };

    explicit PbrPass(Config cfg);

    void build(FrameGraph &fg) const;

private:
    Config m_cfg;
};

}  // namespace lr
