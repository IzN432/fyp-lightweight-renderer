#include "VertexCentroid.hpp"

#include "core/editor/VertexManager.hpp"
#include "core/scene/TransformComponent.hpp"

#include <glm/vec4.hpp>

namespace lr
{

glm::vec3 worldVertexCentroid(const VertexManager &vertices, const TransformComponent &transform,
                              const std::unordered_set<uint32_t> &indices)
{
    if (indices.empty())
    {
        return glm::vec3(transform.worldMatrix()[3]);
    }

    const std::vector<glm::vec3> &positions = vertices.getPositions();
    glm::vec3                     localCentroid(0.0f);
    for (uint32_t index : indices)
    {
        localCentroid += positions[index];
    }
    localCentroid /= static_cast<float>(indices.size());
    return glm::vec3(transform.worldMatrix() * glm::vec4(localCentroid, 1.0f));
}

} // namespace lr
