#pragma once

#include "VertexDragHandler.hpp"

#include "core/editor/VertexManager.hpp"
#include "core/editor/selection/SelectionManager.hpp"
#include "core/editor/command/CommandManager.hpp"

namespace lr
{

// Today's gizmo behavior, extracted unchanged: drives the current selection, applies drag deltas
// straight to VertexManager, and pushes a TranslatePointsCommand on drag end.
class DefaultVertexDragHandler : public VertexDragHandler
{
public:
    DefaultVertexDragHandler(VertexManager &vertexManager, SelectionManager &selectionManager,
                              CommandManager &commandManager)
        : m_vertexManager(vertexManager), m_selectionManager(selectionManager), m_commandManager(commandManager) {}

    const std::unordered_set<uint32_t> &indices() const override { return m_selectionManager.getSelectedIndices(); }
    void beginDrag() override {}
    void translate(const glm::vec3 &frameDelta) override;
    void endDrag(const glm::vec3 &totalDelta) override;

private:
    VertexManager &m_vertexManager;
    SelectionManager &m_selectionManager;
    CommandManager &m_commandManager;
};

} // namespace lr
