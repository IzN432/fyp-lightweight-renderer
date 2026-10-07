#include "EditorStateController.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

void EditorStateController::registerState(EditorStateDefinition state)
{
    if (state.id.empty())
    {
        throw std::invalid_argument("EditorStateController: state id cannot be empty");
    }
    const std::string id = state.id;
    if (!m_states.emplace(id, std::move(state)).second)
    {
        throw std::logic_error("EditorStateController: duplicate state id: " + id);
    }
}

void EditorStateController::activate(std::string_view id)
{
    const auto found = m_states.find(std::string(id));
    if (found == m_states.end())
    {
        throw std::out_of_range("EditorStateController: unknown state: " + std::string(id));
    }
    if (m_active == id)
    {
        return;
    }

    if (!m_active.empty())
    {
        const EditorStateDefinition &outgoing = m_states.at(m_active);
        if (outgoing.onExit)
        {
            outgoing.onExit();
        }
    }

    m_active = found->first;
    m_onChanged(found->second);
    if (found->second.onEnter)
    {
        found->second.onEnter();
    }
}

const EditorStateDefinition &EditorStateController::active() const
{
    const auto found = m_states.find(m_active);
    if (found == m_states.end())
    {
        throw std::logic_error("EditorStateController: no active state");
    }
    return found->second;
}

void EditorStateController::update(const EditorFrameContext &frame) const
{
    const EditorStateDefinition &state = active();
    if (state.update)
    {
        state.update(frame);
    }
}

GizmoRequest EditorStateController::gizmoRequest(const EditorFrameContext &frame) const
{
    const EditorStateDefinition &state = active();
    return state.gizmoRequest ? state.gizmoRequest(frame) : GizmoRequest{};
}

} // namespace lr
