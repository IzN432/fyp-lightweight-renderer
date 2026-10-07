#pragma once

#include "core/editor/EditorFrameContext.hpp"
#include "core/editor/EditorPresentation.hpp"
#include "core/editor/gizmo/GizmoRequest.hpp"

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace lr
{

// Everything a state contributes: what the frame should look like (presentation) and what the state
// does while it is active (the hooks). Every hook is optional; an unset hook means "nothing to do",
// so a purely presentational state still registers as a single aggregate initializer.
struct EditorStateDefinition
{
    std::string        id;
    EditorPresentation presentation;

    // Called as the state becomes active / stops being active. onExit runs before the incoming
    // state's presentation is published, so it still observes the outgoing state's selection;
    // onEnter runs after, so it observes the new presentation.
    std::function<void()> onEnter;
    std::function<void()> onExit;

    // Per-frame work the state owns, e.g. a feature's overlay GUI.
    std::function<void(const EditorFrameContext &)> update;

    // The state's bid for the single shared gizmo. Returning a default-constructed GizmoRequest
    // (or leaving the hook unset) means the state wants no gizmo this frame.
    std::function<GizmoRequest(const EditorFrameContext &)> gizmoRequest;
};

// Registry-backed state controller. Feature code can add states without extending a central enum;
// activating a state atomically publishes its complete rendering and selection policy, and routes
// per-frame behavior to it, so the host never branches on which state is active.
class EditorStateController
{
public:
    using ChangedCallback = std::function<void(const EditorStateDefinition &)>;

    explicit EditorStateController(ChangedCallback onChanged) : m_onChanged(std::move(onChanged)) {}

    void registerState(EditorStateDefinition state);
    void activate(std::string_view id);

    bool                         isActive(std::string_view id) const { return m_active == id; }
    const EditorStateDefinition &active() const;

    // Behavior dispatch. These exist so callers drive "the active state" rather than inspecting
    // which state that is and whether it happens to define the hook.
    void         update(const EditorFrameContext &frame) const;
    GizmoRequest gizmoRequest(const EditorFrameContext &frame) const;

private:
    std::unordered_map<std::string, EditorStateDefinition> m_states;
    std::string                                            m_active;
    ChangedCallback                                        m_onChanged;
};

} // namespace lr
