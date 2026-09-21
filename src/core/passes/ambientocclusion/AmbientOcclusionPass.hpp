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
        float       sphereRadius = 0.5f;
        int         numSteps     = 16;
        int         numDirs      = 8;
        float       tanAngleBias = 0.364f; // tan(20 degrees)
        float       aoScalar     = 2.0f;
    };

    explicit AmbientOcclusionPass(Config cfg);

    // Live-editable — e.g. from an ImGui pane (see main.cpp's "HBAO" window). Call updateParams()
    // after changing anything here for the change to reach the GPU.
    Config       &config() { return m_cfg; }
    const Config &config() const { return m_cfg; }

    // Upload static resources (AO params buffer, output images). Call once before build().
    void uploadResources(ResourceRegistry &resources) const;

    // Re-packs config() into the AO params UBO and pushes it to the GPU — call after editing
    // config() (e.g. from a GUI slider) for the change to take effect. Cheap enough to call every
    // frame while a tuning UI is open; uploadResources() must have run first.
    void updateParams(ResourceRegistry &resources) const;

    // Add the HBAO compute pass plus its bilateral blur to the frame graph.
    // Reads "gbufferDepth" and "gbufferNormal"; writes "hbao_ao" (final, blurred).
    void build(FrameGraph &fg) const;

private:
    Config m_cfg;
};

} // namespace lr
