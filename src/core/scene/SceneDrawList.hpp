#pragma once

#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"
#include "core/upload/MeshUploader.hpp"
#include "core/upload/SkinUploader.hpp"

#include <glm/mat4x4.hpp>

#include <cstdint>
#include <vector>

namespace lr
{

// One mesh's worth of draw state, resolved for the current frame.
struct SceneDraw
{
    glm::mat4 model{1.0f};
    uint32_t  primitiveIdOffset = 0;
    uint32_t  paletteOffset     = 0;
    uint32_t  skinEnabled       = 0;

    uint32_t indexCount   = 0;
    uint32_t firstIndex   = 0;
    int32_t  vertexOffset = 0;
};

// The per-mesh draw state every geometry-replaying pass needs: where each mesh lives in the shared
// vertex/index buffers, the transform to draw it with, the object it came from, and its skinning
// palette. GeometryPass, ObjectPickingPass, TransparentPass and the three shadow passes all walk the
// same list in the same order, so they share one type rather than five parallel vectors each.
//
// Draw indices are identity: ObjectPickingPass writes `index + 1` as its picking ID and the editor
// maps it back through SceneGpu::geometryObjects(), so every pass must agree on ordering.
//
// Transforms and objects are read when the pass executes, not when the list is built, so dragging a
// TransformComponent moves the mesh immediately with no buffer or pass rebuild.
class SceneDrawList
{
public:
    SceneDrawList() = default;

    // Throws std::invalid_argument unless all five arrays are parallel.
    SceneDrawList(VertexBufferUploadResult vertices, IndexBufferUploadResult indices,
                  std::vector<const TransformComponent *> transforms, std::vector<SceneObject *> objects,
                  std::vector<SkinDrawInfo> skins);

    size_t size() const { return m_vertices.singleMeshResults.size(); }

    // False once the draw's object has been removed from its scene; the pass skips it rather than
    // waiting for the next geometry rebuild to repack the buffers.
    bool isLive(size_t i) const;

    // A mesh with no transform has its positions already baked into world space (e.g.
    // AreaLightVisual's quads), and gets an identity model matrix rather than being
    // double-transformed. `skinningEnabled` is the pass's own global toggle, ANDed with the mesh's
    // own skinEnabled.
    SceneDraw at(size_t i, bool skinningEnabled = true) const;

private:
    VertexBufferUploadResult                m_vertices;
    IndexBufferUploadResult                 m_indices;
    std::vector<const TransformComponent *> m_transforms;
    std::vector<SceneObject *>              m_objects;
    std::vector<SkinDrawInfo>               m_skins;
};

} // namespace lr
