#pragma once

#include "core/framegraph/FrameGraph.hpp"

#include <string>

namespace lr
{

class AmbientOcclusionPass
{
public:
    struct Config
    {
        std::string cameraBufferResourceName;
        float sphereRadius = 0.1f;
        int   numSteps     = 32;
        int   numDirs      = 32;
        float tanAngleBias = 0.364f;  // tan(20 degrees)
        float aoScalar     = 2.0f;
    };

    explicit AmbientOcclusionPass(Config cfg);

    // Upload static resources (direction texture, AO params buffer, output images).
    // Call once before build().
    void uploadResources(ResourceRegistry &resources) const;

    // Add the HBAO and blur passes to the frame graph.
    // Reads "gbufferDepth" and "gbufferNormal"; writes "hbao_ao".
    void build(FrameGraph &fg) const;

private:
    Config m_cfg;
};

}  // namespace lr
