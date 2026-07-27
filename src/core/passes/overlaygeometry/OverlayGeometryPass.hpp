#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/passes/overlaygeometry/OverlayInstance.hpp"
#include "core/scene/Mesh.hpp"
#include "core/upload/MeshUploader.hpp"

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <vector>

namespace lr
{

class OverlayGeometryPass
{
public:
    struct Config
    {
        std::string cameraBufferResourceName;
        std::string vertexBufferName       = "overlayMeshVertexBuffer";
        std::string indexBufferName        = "overlayMeshIndexBuffer";
        std::string pickingImageName       = "gizmoPicking";
    };

    explicit OverlayGeometryPass(Config cfg);

    void uploadResources(ResourceRegistry &registry);
    void build(FrameGraph &fg);

    void setInstances(std::vector<OverlayInstance> instances);

    // Set the pickingId (OverlayInstance::pickingId) of the instance currently under the
    // cursor. Use ~0u for "none". The hovered instance is drawn with a lightened color.
    void setHoveredInstance(uint32_t pickingId) { m_hoveredInstance = pickingId; }

    const std::string &pickingImageName() const { return m_cfg.pickingImageName; }

private:
    Config                       m_cfg;
    VertexBufferUploadResult     m_vertexUpload;
    IndexBufferUploadResult      m_indexUpload;
    GpuMeshLayout                m_gpuMeshLayout;
    std::vector<OverlayInstance> m_instances;
    uint32_t                     m_hoveredInstance = ~0u;
};

}  // namespace lr
