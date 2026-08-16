#include "ArapDeformCommand.hpp"

namespace lr
{

void ArapDeformCommand::execute()
{
    std::vector<uint32_t> indices;
    std::vector<glm::vec3> positions;
    indices.reserve(m_diffs.size());
    positions.reserve(m_diffs.size());
    for (const auto &diff : m_diffs)
    {
        indices.push_back(diff.index);
        positions.push_back(diff.after);
    }
    m_vertexManager.setPositions(indices, positions);
}

void ArapDeformCommand::undo()
{
    std::vector<uint32_t> indices;
    std::vector<glm::vec3> positions;
    indices.reserve(m_diffs.size());
    positions.reserve(m_diffs.size());
    for (const auto &diff : m_diffs)
    {
        indices.push_back(diff.index);
        positions.push_back(diff.before);
    }
    m_vertexManager.setPositions(indices, positions);
}

} // namespace lr
