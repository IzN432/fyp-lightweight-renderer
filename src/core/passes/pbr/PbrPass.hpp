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
        std::string cameraBufferResourceName;
        std::string lightBufferResourceName;
        uint32_t    numLights;
        uint32_t    pfMips;
        // Composable users may omit SpotShadowPass; uploadResources then creates disabled fallbacks.
        std::string shadowImageResourceName        = "pbrFallbackSpotShadowMap";
        std::string shadowParamsBufferResourceName = "pbrFallbackSpotShadowParams";
        std::string cascadedShadowImageResourceName = "pbrFallbackCascadedShadowMap";
        std::string cascadedShadowParamsBufferResourceName = "pbrFallbackCascadedShadowParams";
        std::string areaShadowImageResourceName = "pbrFallbackAreaShadowMap";
        std::string areaShadowParamsBufferResourceName = "pbrFallbackAreaShadowParams";
    };

    explicit PbrPass(Config cfg);

    // Takes effect from the next frame. The pass reads this object every frame, so it must outlive the
    // frame graph's use of the pass.
    void setNumLights(uint32_t numLights) { m_cfg.numLights = numLights; }

    // Upload the LTC lookup tables used for area light shading (ltc1, ltc2).
    // Call once before build().
    void uploadResources(ResourceRegistry &resources) const;

    void build(FrameGraph &fg) const;

private:
    Config m_cfg;
};

} // namespace lr
