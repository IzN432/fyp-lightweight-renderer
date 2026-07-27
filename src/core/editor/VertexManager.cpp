#include "VertexManager.hpp"

namespace lr
{

void VertexManager::updatePosition(uint32_t index, const glm::vec3 &newPosition)
{
    m_positions[index] = newPosition;
    m_updateCallback();
}

void VertexManager::removeVertex(uint32_t index)
{
    if (index < m_positions.size())
    {
        m_positions.erase(m_positions.begin() + index);
    }
    m_updateCallback();
}

void VertexManager::addVertex(const glm::vec3 &position)
{
    m_positions.push_back(position);
    m_updateCallback();
}

void VertexManager::translateSelectedVertices(const std::vector<uint32_t> &indices, const glm::vec3 &translation)
{
    for (uint32_t index : indices)
    {
        if (index < m_positions.size())
        {
            m_positions[index] += translation;
        }
    }
    m_updateCallback();
}

} // namespace lr