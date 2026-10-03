#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/TransformComponent.hpp"
#include "core/upload/MeshUploader.hpp"

#include <vulkan/vulkan.h>

namespace lr
{

// Rasterises the selected mesh with its per-vertex "heatmapColors" attribute (see SceneManager's
// heatmap buffer) Gouraud-interpolated across the surface, into its own color+depth target —
// a debug/analysis view showing scalar-derived colors written per vertex. FinalPass blends
// the result on top of the lit scene when enabled. Disabled by default; toggled at runtime
// via setEnabled().
class HeatmapPass
{
public:
    struct Config
    {
        std::string cameraBufferResourceName;
        std::string
            vertexBufferResourceName; // selected mesh position + color, in the corner domain
        std::string
            indexBufferResourceName; // shared scene index buffer

        VertexBufferUploadResult vertexBufferUploadResult;
        IndexBufferUploadResult  indexBufferUploadResult;

        // Read fresh every frame, same as GeometryPass::Config::meshTransforms — dragging the
        // mesh's TransformComponent moves the heatmap surface immediately with no pass rebuild.
        const TransformComponent *meshTransform = nullptr;
    };

    explicit HeatmapPass(Config cfg);

    void build(FrameGraph &fg, const GpuMeshLayout &layout) const;

    void setEnabled(bool e) { m_enabled = e; }
    bool isEnabled() const { return m_enabled; }

    void setMeshSource(VertexBufferUploadResult vertexUpload, IndexBufferUploadPerMeshResult indexRange,
                       const TransformComponent &transform);

private:
    Config       m_cfg;
    IndexBufferUploadPerMeshResult m_indexRange;
    mutable bool m_enabled = false;
};

} // namespace lr
