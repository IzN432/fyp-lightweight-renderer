#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/scene/Mesh.hpp"
#include "core/upload/MeshUploader.hpp"

#include <vulkan/vulkan.h>

namespace lr
{

class OverlayPointsPass
{
public:
    struct Config
    {
        std::string              cameraBufferResourceName;
        std::string              positionBufferResourceName;
        VertexBufferUploadResult positionBufferUploadResult;
        std::vector<uint32_t>    vertexCounts;
        float                    pointSize       = 2.0f;
        float                    occludedOpacity = 0.5f;
    };

    explicit OverlayPointsPass(Config cfg);

    void build(FrameGraph &fg, const GpuMeshLayout &layout) const;

private:
    Config m_cfg;
};

}  // namespace lr
