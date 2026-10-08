#include "core/scene/SceneDrawList.hpp"

#include "core/scene/Scene.hpp"
#include "features/linear_blend_skinning/Skin.hpp"

#include <stdexcept>

namespace lr
{

SceneDrawList::SceneDrawList(VertexBufferUploadResult vertices, IndexBufferUploadResult indices,
                             std::vector<const TransformComponent *> transforms, std::vector<SceneObject *> objects,
                             std::vector<SkinDrawInfo> skins)
{
    const size_t count = vertices.singleMeshResults.size();
    if (indices.singleMeshResults.size() != count || transforms.size() != count || objects.size() != count ||
        skins.size() != count)
    {
        throw std::invalid_argument("SceneDrawList: draw arrays must be parallel");
    }
    m_vertices   = std::move(vertices);
    m_indices    = std::move(indices);
    m_transforms = std::move(transforms);
    m_objects    = std::move(objects);
    m_skins      = std::move(skins);
}

bool SceneDrawList::isLive(size_t i) const
{
    const SceneObject *object = m_objects[i];
    return object != nullptr && object->scene().contains(object->id());
}

SceneDraw SceneDrawList::at(size_t i, bool skinningEnabled) const
{
    const auto               &mesh      = m_vertices.singleMeshResults[i];
    const auto               &range     = m_indices.singleMeshResults[i];
    const TransformComponent *transform = m_transforms[i];
    const SkinDrawInfo       &skin      = m_skins[i];
    return SceneDraw{
        .model             = transform ? transform->worldMatrix() : glm::mat4(1.0f),
        .primitiveIdOffset = range.firstIndex / 3,
        .paletteOffset     = skin.paletteOffset,
        .skinEnabled       = skin.skin && skin.skin->skinningEnabled() && skinningEnabled ? 1u : 0u,
        .indexCount        = range.indexCount,
        .firstIndex        = range.firstIndex,
        .vertexOffset      = static_cast<int32_t>(mesh.vertexOffset),
    };
}

} // namespace lr
