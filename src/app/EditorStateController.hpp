#pragma once

#include "core/editor/EditorPresentation.hpp"

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace lr
{

struct EditorStateDefinition
{
    std::string        id;
    EditorPresentation presentation;
};

// Registry-backed state controller. Feature code can add states without extending a central enum;
// activating a state atomically publishes its complete rendering and selection policy.
class EditorStateController
{
public:
    using ChangedCallback = std::function<void(const EditorStateDefinition &)>;

    explicit EditorStateController(ChangedCallback onChanged) : m_onChanged(std::move(onChanged)) {}

    void registerState(EditorStateDefinition state);
    void activate(std::string_view id);

    bool isActive(std::string_view id) const { return m_active == id; }
    const EditorStateDefinition &active() const;

private:
    std::unordered_map<std::string, EditorStateDefinition> m_states;
    std::string                                             m_active;
    ChangedCallback                                         m_onChanged;
};

} // namespace lr
