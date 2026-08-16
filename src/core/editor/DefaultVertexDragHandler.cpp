#include "DefaultVertexDragHandler.hpp"

#include "core/editor/command/TranslatePointsCommand.hpp"

#include <memory>

namespace lr
{

void DefaultVertexDragHandler::translate(const glm::vec3 &frameDelta)
{
    m_vertexManager.translateSelectedVertices(m_selectionManager.getSelectedIndices(), frameDelta);
}

void DefaultVertexDragHandler::endDrag(const glm::vec3 &totalDelta)
{
    m_commandManager.appendCommandWithoutExecuting(std::make_unique<TranslatePointsCommand>(
        m_vertexManager, m_selectionManager.getSelectedIndices(), totalDelta));
}

} // namespace lr
