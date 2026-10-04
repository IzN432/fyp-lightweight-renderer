#pragma once

#include "core/framegraph/FrameGraph.hpp"

#include <vulkan/vulkan.h>

#include <string>

namespace lr
{

// The engine's final image without the editor overlays FinalPass blends in: the environment cubemap
// where GeometryPass drew nothing, the HDR input image elsewhere, Reinhard tone mapped.
//
// Reads:  cameraBufferResourceName (CameraGpuData UBO), "ibl_env" (cubemap, from IBLPass),
//         "gbufferDepth" (D32, from GeometryPass), inputImage (HDR colour, default "pbr").
// Writes: outputImage (default "swapchain") in outputFormat.
class CompositePass
{
public:
    struct Config
    {
        std::string cameraBufferResourceName;
        VkFormat    outputFormat;
        std::string inputImage  = "pbr";
        std::string outputImage = "swapchain";
        std::string passName    = "composite";
    };

    explicit CompositePass(Config cfg);

    void build(FrameGraph &fg) const;

private:
    Config m_cfg;
};

} // namespace lr
