#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/scene/Mesh.hpp"
#include "core/upload/MeshUploader.hpp"

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <vector>

namespace lr
{

enum class OverlayPrimitive : uint32_t
{
    Cube   = 0,
    Sphere = 1,
    Arrow  = 2,
};

struct OverlayInstance
{
    OverlayPrimitive primitive       = OverlayPrimitive::Cube;
    glm::vec3        position        = {0.0f, 0.0f, 0.0f};
    glm::vec3        eulerDegrees    = {0.0f, 0.0f, 0.0f}; // GLM YXZ intrinsic order
    glm::vec3        scale           = {1.0f, 1.0f, 1.0f};
    glm::vec3        color           = {1.0f, 0.0f, 1.0f};
    float            occludedOpacity = 0.0f;
};

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

    // Set the 0-based index of the instance currently under the cursor.
    // Use ~0u for "none". The hovered instance is drawn with a lightened color.
    void setHoveredInstance(uint32_t idx) { m_hoveredInstance = idx; }

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
