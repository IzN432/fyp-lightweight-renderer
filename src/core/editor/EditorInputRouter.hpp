#pragma once

#include "core/editor/EditorPresentation.hpp"

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace lr
{

// A pointer-button press or release as the editor sees it, with the modifier state at the time.
struct PointerButtonEvent
{
    int  button;
    int  action;
    bool shift = false;
    bool ctrl  = false;
    bool alt   = false;
};

// Immutable editor policy accompanying a routed input event. Layers should prefer presentation
// capabilities over state-name checks; activeState remains available when behavior truly belongs
// to one specific state.
struct EditorInputContext
{
    std::string_view   activeState;
    EditorPresentation presentation;

    bool isState(std::string_view id) const { return activeState == id; }
};

// The editor's pointer priority, in one place. Button events descend through registered layers in
// priority order and stop at the first one that consumes them; nothing is offered to any layer while
// the UI or the active gizmo owns the pointer.
//
// Camera navigation is deliberately not a layer. It reads pointer *state* each frame rather than
// consuming events, and it answers to a different rule — see viewportNavigationAllowed().
class EditorInputRouter
{
public:
    using CaptureQuery = std::function<bool()>;
    using ButtonLayer  = std::function<bool(const PointerButtonEvent &, const EditorInputContext &)>;

    // The two things that outrank every layer: the UI, and the active gizmo's handles.
    EditorInputRouter(CaptureQuery uiCapturesPointer, CaptureQuery gizmoCapturesPointer);

    // Layers are offered events in registration order, so register from highest priority down.
    void addButtonLayer(std::string name, ButtonLayer layer);

    // Returns whether a layer consumed the event.
    bool routeButton(const PointerButtonEvent &event, const EditorInputContext &context) const;

    // Whether the UI or the active gizmo currently owns the pointer, which is what stops button
    // events reaching any layer.
    bool pointerCaptured() const;

    // Camera orbit/pan/zoom. A hovered gizmo raises the UI's pointer-capture flag — ImGuizmo routes
    // its handles through ImGui — but navigation uses buttons the gizmo never takes, so only real UI
    // should stop it. That is why this is not simply "nothing above me is capturing".
    bool viewportNavigationAllowed() const;

private:
    struct Layer
    {
        std::string name;
        ButtonLayer handler;
    };

    CaptureQuery       m_uiCapturesPointer;
    CaptureQuery       m_gizmoCapturesPointer;
    std::vector<Layer> m_layers;
};

} // namespace lr
