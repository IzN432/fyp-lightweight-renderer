#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/Transform.hpp"
#include "core/upload/MeshUploader.hpp"

#include <vulkan/vulkan.h>

namespace lr
{

// Rasterises the main mesh with its per-vertex "heatmapColors" attribute (see SceneManager's
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
            vertexBufferResourceName; // position + color, corner domain (see SceneManager::mainMeshHeatmapBufferName)
        std::string
            indexBufferResourceName; // shared mesh index buffer — only singleMeshResults[0] (the main mesh) is drawn

        VertexBufferUploadResult vertexBufferUploadResult;
        IndexBufferUploadResult  indexBufferUploadResult;

        // Read fresh every frame, same as GeometryPass::Config::meshTransforms — dragging the
        // mesh's Transform moves the heatmap surface immediately with no pass rebuild.
        const Transform *meshTransform = nullptr;
    };

    explicit HeatmapPass(Config cfg);

    void build(FrameGraph &fg, const GpuMeshLayout &layout) const;

    void setEnabled(bool e) { m_enabled = e; }
    bool isEnabled() const { return m_enabled; }

private:
    Config       m_cfg;
    mutable bool m_enabled = false;
};

} // namespace lr
