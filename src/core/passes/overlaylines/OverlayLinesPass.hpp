#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/passes/overlaylines/OverlayLine.hpp"

#include <string>
#include <vector>

namespace lr
{

class OverlayLinesPass
{
public:
    struct Config
    {
        std::string cameraBufferResourceName;
        std::string vertexBufferResourceName = "overlayLineVertices";
        uint32_t    maxLineCount              = 65536;
    };

    OverlayLinesPass(Config config, ResourceRegistry &registry);

    void build(FrameGraph &frameGraph);
    void setLines(const std::vector<OverlayLine> &lines);

private:
    struct GpuVertex
    {
        glm::vec4 positionAndVisibleOpacity;
        glm::vec4 colorAndOccludedOpacity;
    };

    Config            m_config;
    ResourceRegistry *m_registry;
    uint32_t          m_vertexCount = 0;
};

} // namespace lr
