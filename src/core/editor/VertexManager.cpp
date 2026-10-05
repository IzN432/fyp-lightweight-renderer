#include "VertexManager.hpp"

#include <algorithm>

namespace lr
{

void VertexManager::updatePosition(uint32_t index, const glm::vec3 &newPosition)
{
    m_mesh->setPositionAt(index, newPosition);
}

void VertexManager::translateSelectedVertices(const std::unordered_set<uint32_t> &indices, const glm::vec3 &translation)
{
    std::vector<uint32_t> changedIndices;
    std::vector<glm::vec3> positions;
    changedIndices.reserve(indices.size());
    positions.reserve(indices.size());
    for (uint32_t index : indices)
    {
        if (index < m_mesh->uniquePositionCount())
        {
            changedIndices.push_back(index);
            positions.push_back(m_mesh->positionAt(index) + translation);
        }
    }
    m_mesh->setPositions(changedIndices, positions);
}

void VertexManager::setPositions(const std::vector<uint32_t> &indices, const std::vector<glm::vec3> &positions)
{
    std::vector<uint32_t> changedIndices;
    std::vector<glm::vec3> changedPositions;
    const size_t count = std::min(indices.size(), positions.size());
    changedIndices.reserve(count);
    changedPositions.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        if (indices[i] < m_mesh->uniquePositionCount())
        {
            changedIndices.push_back(indices[i]);
            changedPositions.push_back(positions[i]);
        }
    }
    m_mesh->setPositions(changedIndices, changedPositions);
}

} // namespace lr
