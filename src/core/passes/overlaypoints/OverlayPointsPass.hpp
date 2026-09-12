#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/TransformComponent.hpp"
#include "core/upload/MeshUploader.hpp"

#include <vulkan/vulkan.h>

namespace lr
{

class OverlayPointsPass
{
public:
    struct Config
    {
        std::string cameraBufferResourceName;
        // Interleaved deduped position + color buffer (see SceneManager::mainMeshPointsBufferName).
        std::string              pointsBufferResourceName;
        VertexBufferUploadResult pointsBufferUploadResult;
        std::vector<uint32_t>    vertexCounts;

        // The mesh's TransformComponent, read fresh every frame so the points stay aligned with the mesh
        // as it moves — same reasoning as GeometryPass::Config::meshTransforms. Null means draw
        // with an identity model matrix.
        const TransformComponent *meshTransform = nullptr;
    };

    explicit OverlayPointsPass(Config cfg);

    void build(FrameGraph &fg, const GpuMeshLayout &layout) const;

    void setEnabled(bool e) { m_enabled = e; }

private:
    Config       m_cfg;
    mutable bool m_enabled = true;
};

} // namespace lr
