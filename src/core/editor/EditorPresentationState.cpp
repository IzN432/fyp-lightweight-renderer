#include "EditorPresentationState.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

EditorPresentationState::EditorPresentationState(std::function<void()> onVertexSelectionWithdrawn)
    : m_onVertexSelectionWithdrawn(std::move(onVertexSelectionWithdrawn))
{
    if (!m_onVertexSelectionWithdrawn)
    {
        throw std::invalid_argument("EditorPresentationState: withdrawal callback cannot be empty");
    }
}

void EditorPresentationState::set(EditorPresentation presentation)
{
    if (presentation == m_current)
    {
        return;
    }

    const bool selectionWasActive = m_current.vertexSelectionActive;
    m_current                     = presentation;

    // Invalidate before notifying, so no observer of the new presentation can read a selection that
    // the change has already made meaningless.
    if (selectionWasActive && !m_current.vertexSelectionActive)
    {
        m_onVertexSelectionWithdrawn();
    }

    m_changedCallbacks.invoke(m_current);
}

CallbackConnection
EditorPresentationState::registerChangedCallback(std::function<void(const EditorPresentation &)> callback)
{
    return m_changedCallbacks.connect(std::move(callback));
}

} // namespace lr
