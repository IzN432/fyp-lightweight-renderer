#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/scene/Mesh.hpp"
#include "core/upload/MeshUploader.hpp"

#include <vulkan/vulkan.h>

namespace lr
{

class OverlayGeometryPass
{
public:
    struct Config
    {
        std::string cameraBufferResourceName;
        std::string vertexBufferName = "overlayMeshVertexBuffer";
        std::string indexBufferName  = "overlayMeshIndexBuffer";
    };

    explicit OverlayGeometryPass(Config cfg);

    void uploadResources(ResourceRegistry &registry, const std::vector<const Mesh*> &meshes);
    void build(FrameGraph &fg) const;

private:
    Config m_cfg;
    VertexBufferUploadResult m_vertexUpload;
    IndexBufferUploadResult  m_indexUpload;
    GpuMeshLayout            m_gpuMeshLayout;
};

}  // namespace lr
