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
        const uint32_t         *numLights; // read each frame (see PbrPass::build) so lights added at runtime are picked up without rebuilding this pass
        uint32_t                pfMips;
    };

    explicit PbrPass(Config cfg);

    // Upload the LTC lookup tables used for area light shading (ltc1, ltc2).
    // Call once before build().
    void uploadResources(ResourceRegistry &resources) const;

    void build(FrameGraph &fg) const;

private:
    Config m_cfg;
};

}  // namespace lr
