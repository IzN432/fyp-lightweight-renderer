#pragma once

#include "core/editor/EditorPresentation.hpp"
#include "core/utility/CallbackList.hpp"

#include <functional>

namespace lr
{

// The live editor presentation, plus the policy that follows from changing it.
//
// Kept apart from whoever applies the flags to the renderer, because the interesting part is not the
// presentation but the *transition*: withdrawing a capability invalidates state that was only
// meaningful while that capability was available. Keeping that rule here — rather than inside a
// class that owns GPU resources — is what makes it testable on its own.
class EditorPresentationState
{
public:
    // `onVertexSelectionWithdrawn` is called when a change takes vertex selection away, meaning the
    // current selection has stopped referring to anything: vertex indices are only meaningful while
    // a state actually selects vertices. Passed as a callback so this class needs to know nothing
    // about who holds the selection or when that owner comes into existence.
    explicit EditorPresentationState(std::function<void()> onVertexSelectionWithdrawn);

    const EditorPresentation &current() const { return m_current; }

    // Setting the presentation already in effect does nothing at all: an identical presentation is
    // not a transition, so it neither notifies nor invalidates. That is what lets two states wanting
    // the same capabilities — plain vertex editing and ARAP — hand a selection between them.
    void set(EditorPresentation presentation);

    CallbackConnection registerChangedCallback(std::function<void(const EditorPresentation &)> callback);

private:
    EditorPresentation                       m_current;
    std::function<void()>                    m_onVertexSelectionWithdrawn;
    CallbackList<const EditorPresentation &> m_changedCallbacks;
};

} // namespace lr
