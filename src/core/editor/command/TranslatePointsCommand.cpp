#include "TranslatePointsCommand.hpp"

namespace lr
{

void TranslatePointsCommand::execute()
{
    // Implementation of the execute method
    m_vertexManager.translateSelectedVertices(m_indices, m_translation);
}

void TranslatePointsCommand::undo()
{
    // Implementation of the undo method
    m_vertexManager.translateSelectedVertices(m_indices, -m_translation);
}

}