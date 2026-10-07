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
    m_active = found->first;
    m_onChanged(found->second);
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

} // namespace lr
