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
    static constexpr const char *imageName = "gizmoPicking";

    struct Config
    {
        std::string cameraBufferResourceName;
        std::string vertexBufferName = "overlayMeshVertexBuffer";
        std::string indexBufferName  = "overlayMeshIndexBuffer";
    };

    explicit OverlayGeometryPass(Config cfg);

    void uploadResources(ResourceRegistry &registry);
    void build(FrameGraph &fg);

    void setInstances(std::vector<OverlayInstance> instances);

    // Set the pickingId (OverlayInstance::pickingId) of the instance currently under the
    // cursor. Use ~0u for "none". The hovered instance is drawn brighter (same hue and saturation).
    void setHoveredInstance(uint32_t pickingId) { m_hoveredInstance = pickingId; }

private:
    Config                       m_cfg;
    VertexBufferUploadResult     m_vertexUpload;
    IndexBufferUploadResult      m_indexUpload;
    GpuMeshLayout                m_gpuMeshLayout;
    std::vector<OverlayInstance> m_instances;
    uint32_t                     m_hoveredInstance = ~0u;
};

} // namespace lr
