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
        // Interleaved deduped position + color buffer for the selected mesh.
        std::string              pointsBufferResourceName;
        VertexBufferUploadResult pointsBufferUploadResult;
        std::vector<uint32_t>    vertexCounts;

        // The mesh's TransformComponent, read fresh every frame so the points stay aligned with the mesh
        // as it moves — same reasoning as SceneDrawList::transforms(). Null means draw
        // with an identity model matrix.
        const TransformComponent *meshTransform = nullptr;
    };

    explicit OverlayPointsPass(Config cfg);

    void build(FrameGraph &fg, const GpuMeshLayout &layout) const;

    void setEnabled(bool e) { m_enabled = e; }

    // Repoints the pass at a different mesh's points buffer contents — the buffer resource itself
    // (pointsBufferResourceName) is reused in place (see ResourceRegistry::replaceUploadedBuffer),
    // so only the per-mesh draw bookkeeping needs updating here. Config always describes exactly
    // one mesh (see the single-entry vectors above), so this replaces them wholesale rather than
    // appending.
    void setPointsSource(VertexBufferUploadResult uploadResult, uint32_t vertexCount,
                         const TransformComponent &meshTransform)
    {
        m_cfg.pointsBufferUploadResult = std::move(uploadResult);
        m_cfg.vertexCounts             = {vertexCount};
        m_cfg.meshTransform            = &meshTransform;
    }

private:
    Config       m_cfg;
    mutable bool m_enabled = true;
};

} // namespace lr
