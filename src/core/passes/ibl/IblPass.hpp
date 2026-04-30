#pragma once

#include "core/framegraph/FrameGraph.hpp"

#include <filesystem>

namespace lr
{

class IBLPass
{
public:
    struct Config
    {
        std::filesystem::path hdriPath;
        uint32_t envRes = 2048;
        uint32_t irrRes = 32;
        uint32_t pfRes  = 512;
        uint32_t pfMips = 10;
    };

    explicit IBLPass(Config cfg);

    // Upload the HDRI and register output cubemaps into the resource registry.
    void uploadResources(ResourceRegistry &resources) const;

    // Add all IBL compute passes to fg and execute them once.
    // The pass graph is cleared afterwards; the registry keeps the results.
    void preprocess(FrameGraph &fg) const;

private:
    Config m_cfg;
};

}  // namespace lr
