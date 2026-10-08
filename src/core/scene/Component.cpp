#include "core/scene/Component.hpp"

#include "core/editor/EditorContext.hpp"

#include <imgui.h>

#include <memory>
#include <typeindex>

namespace lr
{
namespace
{
constexpr float kDisabledMenuTextAlpha = 0.40f;
} // namespace

void Component::onGUI(EditorContext &context)
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
        ImGui::Text("Component: %s", m_name.c_str());
        onGUIImpl();
        drawValueClipboardMenu(context);
    }
    ImGui::EndChild();
    m_editorContext = nullptr;
}

void Component::drawValueClipboardMenu(EditorContext &context)
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

    ImGui::EndPopup();
}

} // namespace lr
