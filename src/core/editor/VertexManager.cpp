#include "VertexManager.hpp"

#include <algorithm>

namespace lr
{

void VertexManager::notifyUpdateCallbacks()
{
    for (const auto &callback : m_updateCallbacks)
    {
        callback();
    }
}

void VertexManager::updatePosition(uint32_t index, const glm::vec3 &newPosition)
{
    m_mesh->positionAt(index) = newPosition;
    notifyUpdateCallbacks();
}

void VertexManager::translateSelectedVertices(const std::unordered_set<uint32_t> &indices, const glm::vec3 &translation)
{
    for (uint32_t index : indices)
    {
        if (index < m_mesh->uniquePositionCount())
        {
            m_mesh->positionAt(index) += translation;
        }
    }
    notifyUpdateCallbacks();
}

void VertexManager::setPositions(const std::vector<uint32_t> &indices, const std::vector<glm::vec3> &positions)
{
    const size_t count = std::min(indices.size(), positions.size());
    for (size_t i = 0; i < count; ++i)
    {
        if (indices[i] < m_mesh->uniquePositionCount())
        {
            m_mesh->positionAt(indices[i]) = positions[i];
        }
    }
    notifyUpdateCallbacks();
}

} // namespace lr
