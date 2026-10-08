#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/passes/geometry/GeometryPass.hpp"

namespace lr
{

// Replays scene geometry into a single-sample uint target. IDs are dense draw indices; the editor
// maps them back to stable SceneObjectIds using the same geometryObjects array.
class ObjectPickingPass
{
public:
    static constexpr const char *imageName = "objectPicking";

    explicit ObjectPickingPass(GeometryPass::Config config) : m_config(std::move(config)) {}

    void uploadResources(ResourceRegistry &resources) const;
    void build(FrameGraph &frameGraph, const GpuMeshLayout &layout) const;
    void setSceneGeometry(VertexBufferUploadResult vertices, IndexBufferUploadResult indices,
                          std::vector<const TransformComponent *> transforms, std::vector<SceneObject *> objects,
                          std::vector<SkinDrawInfo> skins);

private:
    GeometryPass::Config m_config;
};

} // namespace lr
