#include "DefaultVertexDragHandler.hpp"

#include "core/editor/command/TranslatePointsCommand.hpp"

#include <memory>

namespace lr
{

void DefaultVertexDragHandler::translate(const glm::vec3 &frameDelta)
{
    const glm::vec3 localDelta = worldDeltaToLocal(frameDelta);
    m_vertexManager.translateSelectedVertices(m_selectionManager.getSelectedIndices(), localDelta);
    m_accumulatedLocalDelta += localDelta;
}

void DefaultVertexDragHandler::endDrag(const glm::vec3 &)
{
    if (glm::dot(m_accumulatedLocalDelta, m_accumulatedLocalDelta) == 0.0f)
    {
        return;
    }
    m_commandManager.appendCommandWithoutExecuting(
        std::make_unique<TranslatePointsCommand>(m_vertexManager, m_selectionManager.getSelectedIndices(),
                                                 m_accumulatedLocalDelta));
}

} // namespace lr
