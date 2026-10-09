#include "core/scene/Component.hpp"

#include "core/editor/EditorContext.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/editor/command/EditComponentValuesCommand.hpp"
#include "core/scene/SceneObject.hpp"

#include <imgui.h>

#include <memory>
#include <optional>
#include <typeindex>
#include <utility>

namespace lr
{
namespace
{
constexpr float kDisabledMenuTextAlpha = 0.40f;
} // namespace

void Component::onGUI(EditorContext &context, std::optional<std::type_index> &removalRequest)
{
    m_editorContext = &context;
    // Each component draws into its own bordered child window. That separates the components in the
    // inspector more plainly than a rule between them did, and it gives the right-click menu below
    // a window of its own to belong to. Its ID comes from the PushID SceneObject::onGUI wraps each
    // component in, so sibling components do not share one.
    const bool visible = ImGui::BeginChild("component", ImVec2(0.0f, 0.0f),
                                           ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    if (visible)
    {
        // Whatever this component's inspector records becomes one undo, the edit itself together
        // with anything it set off: a transform committed here with Auto Key on also writes a
        // keyframe, and taking the edit back has to take that with it.
        CommandTransaction transaction(context.commands);

        ImGui::Text("Component: %s", m_name.c_str());
        // Read before the widgets run, so it is the state the frame started in — and compared after
        // the menu, because pasting values over this component is an edit like any other.
        std::unique_ptr<ComponentValues> beforeThisFrame = undoValues();
        onGUIImpl();
        drawComponentMenu(context, removalRequest);
        recordValueEdit(context, std::move(beforeThisFrame), removalRequest.has_value());
    } else
    {
        // Nothing of this component is on screen, so an interaction it was holding cannot still be
        // in progress. Dropping the snapshot keeps a stale one from being compared against values
        // that something other than the inspector has moved in the meantime.
        m_valuesBeforeEdit.reset();
    }
    ImGui::EndChild();
    m_editorContext = nullptr;
}

void Component::recordValueEdit(EditorContext &context, std::unique_ptr<ComponentValues> beforeThisFrame,
                                bool beingRemoved)
{
    // Either the component takes no part in the undo system, or it has no object for a command to
    // name. Nothing to record either way.
    if (!beforeThisFrame || m_owningObject == nullptr)
    {
        return;
    }

    // The component is about to be deleted, and RemoveComponentCommand carries its values away. A
    // value edit recorded here would be undone against a component that is no longer there.
    if (beingRemoved)
    {
        m_valuesBeforeEdit.reset();
        return;
    }

    // Whether a widget that could be editing this component is still being held. ImGui has one
    // active item at a time and a release is always a frame of its own, so waiting for there to be
    // none is what turns a drag reporting a new value every frame into a single edit.
    //
    // The focus test is what keeps a drag elsewhere in the UI from being mistaken for one here;
    // once this component has a snapshot pending, though, the interaction stays its own even while
    // the active item sits in another window — which is where a color picker's popup lives.
    const bool interactionInProgress =
        ImGui::IsAnyItemActive() &&
        (m_valuesBeforeEdit != nullptr || ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows));
    if (interactionInProgress)
    {
        // The first frame of an interaction holds the values it started from; later frames of the
        // same one hold the intermediate values the user is dragging through.
        if (!m_valuesBeforeEdit)
        {
            m_valuesBeforeEdit = std::move(beforeThisFrame);
        }
        return;
    }

    std::unique_ptr<ComponentValues> before =
        m_valuesBeforeEdit ? std::move(m_valuesBeforeEdit) : std::move(beforeThisFrame);
    m_valuesBeforeEdit.reset();

    std::unique_ptr<ComponentValues> after = undoValues();
    if (!after || after->equals(*before))
    {
        return;
    }

    // Appended rather than executed: the inspector's widgets have already applied the edit.
    context.commands.appendCommandWithoutExecuting(std::make_unique<EditComponentValuesCommand>(
        m_owningObject->scene(), m_owningObject->id(), std::type_index(typeid(*this)), std::move(before),
        std::move(after)));
}

void Component::drawComponentMenu(EditorContext &context, std::optional<std::type_index> &removalRequest)
{
    // typeid on a polymorphic reference gives the concrete component type, which is also the key
    // SceneObject stores the component under, so copying and pasting agree on identity.
    const std::type_index type(typeid(*this));

    // The menu belongs to this component's child window, so a right-click anywhere over the
    // component's block opens it, over its widgets included — ImGui's drag widgets claim only the
    // left button.
    if (!ImGui::BeginPopupContextWindow())
    {
        return;
    }

    // Both entries are always listed and disabled rather than hidden, so the menu reads the same
    // way over every component: it names what copy/paste would do here, and shows when neither is
    // available. A component that does not take part leaves both greyed out.
    std::unique_ptr<ComponentValues> snapshot = copyValues();
    ImGui::BeginDisabled(snapshot == nullptr);
    if (ImGui::MenuItem("Copy component values"))
    {
        context.componentClipboard.store(type, std::move(snapshot), valuesAdder());
    }
    ImGui::EndDisabled();

    ImGui::BeginDisabled(!context.componentClipboard.holdsValuesFor(type) || !acceptsPastedValues());
    if (ImGui::MenuItem("Paste component values"))
    {
        pasteValues(*context.componentClipboard.valuesFor(type));
    }
    ImGui::EndDisabled();

    ImGui::Separator();
    ImGui::BeginDisabled(!allowsRemoval());
    if (ImGui::MenuItem("Delete component"))
    {
        removalRequest = type;
    }
    ImGui::EndDisabled();

    ImGui::EndPopup();
}

} // namespace lr
