#include "Scene.hpp"

#include "core/editor/EditorContext.hpp"
#include "core/editor/command/AddComponentCommand.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/editor/command/RenameSceneObjectCommand.hpp"
#include "core/scene/TransformComponent.hpp"

#include <imgui.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <typeindex>
#include <utility>

namespace
{

int resizeStringInput(ImGuiInputTextCallbackData *data)
{
    auto &value = *static_cast<std::string *>(data->UserData);
    value.resize(static_cast<size_t>(data->BufTextLen));
    data->Buf = value.data();
    return 0;
}

bool inputText(const char *label, std::string &value, ImGuiInputTextFlags flags)
{
    flags |= ImGuiInputTextFlags_CallbackResize;
    return ImGui::InputText(label, value.data(), value.capacity() + 1, flags, resizeStringInput,
                            &value);
}

} // namespace

namespace lr
{

SceneObject &Scene::createSceneObject() { return createSceneObject(generateUuid()); }

SceneObject &Scene::createSceneObject(SceneObjectId id)
{
    if (id.is_nil())
    {
        throw std::invalid_argument("A scene object cannot be created with a nil ID");
    }
    if (m_objectIndices.contains(id))
    {
        throw std::invalid_argument("A scene object with ID " + toString(id) + " already exists");
    }
    m_objectIndices.emplace(id, m_sceneObjects.size());
    m_sceneObjects.push_back(std::unique_ptr<SceneObject>(new SceneObject(*this, id)));
    return *m_sceneObjects.back();
}

SceneObject *Scene::find(SceneObjectId id)
{
    const auto found = m_objectIndices.find(id);
    return found == m_objectIndices.end() ? nullptr : m_sceneObjects[found->second].get();
}

const SceneObject *Scene::find(SceneObjectId id) const
{
    const auto found = m_objectIndices.find(id);
    return found == m_objectIndices.end() ? nullptr : m_sceneObjects[found->second].get();
}

SceneObject &Scene::getSceneObject(SceneObjectId id)
{
    if (!contains(id))
    {
        throw std::out_of_range("Scene object ID " + toString(id) + " does not name a live object");
    }
    return *find(id);
}

const SceneObject &Scene::getSceneObject(SceneObjectId id) const
{
    if (!contains(id))
    {
        throw std::out_of_range("Scene object ID " + toString(id) + " does not name a live object");
    }
    return *find(id);
}

bool Scene::contains(SceneObjectId id) const
{
    const SceneObject *object = find(id);
    return object && object->m_alive;
}

bool Scene::canDestroySceneObject(SceneObjectId id) const
{
    if (!contains(id))
    {
        return false;
    }

    std::vector<SceneObjectId> pending{id};
    while (!pending.empty())
    {
        const SceneObjectId current = pending.back();
        pending.pop_back();
        if (m_protectedObjects.contains(current))
        {
            return false;
        }
        const auto &children = find(current)->m_children;
        pending.insert(pending.end(), children.begin(), children.end());
    }
    return true;
}

void Scene::destroySceneObject(SceneObjectId id)
{
    if (!contains(id))
    {
        throw std::out_of_range("Scene object ID is not alive");
    }

    std::vector<SceneObjectId> destroyed;
    std::vector<SceneObjectId> pending{id};
    while (!pending.empty())
    {
        const SceneObjectId current = pending.back();
        pending.pop_back();
        if (m_protectedObjects.contains(current))
        {
            throw std::invalid_argument("A protected scene object cannot be deleted");
        }
        destroyed.push_back(current);
        const auto &children = find(current)->m_children;
        pending.insert(pending.end(), children.begin(), children.end());
    }

    SceneObject &root = *find(id);
    if (root.m_parent && contains(*root.m_parent))
    {
        std::erase(find(*root.m_parent)->m_children, id);
    }
    for (SceneObjectId destroyedId : destroyed)
    {
        SceneObject &object = *find(destroyedId);
        object.m_parent.reset();
        object.m_children.clear();
        object.m_alive = false;
    }
    if (m_selectedObject && std::ranges::find(destroyed, *m_selectedObject) != destroyed.end())
    {
        m_selectedObject.reset();
    }
    if (m_hierarchySelectedObject &&
        std::ranges::find(destroyed, *m_hierarchySelectedObject) != destroyed.end())
    {
        clearHierarchySelection();
    }
    m_objectsDestroyedCallbacks.invoke(destroyed);
}

void Scene::purgeDestroyedSceneObjects()
{
    std::erase_if(m_sceneObjects, [](const std::unique_ptr<SceneObject> &object) {
        return !object->m_alive;
    });
    m_objectIndices.clear();
    for (size_t index = 0; index < m_sceneObjects.size(); ++index)
    {
        m_objectIndices.emplace(m_sceneObjects[index]->id(), index);
    }
}

void Scene::setParent(SceneObjectId childId, std::optional<SceneObjectId> parentId)
{
    SceneObject &child = getSceneObject(childId);
    if (parentId)
    {
        getSceneObject(parentId.value());
        if (parentId.value() == childId)
        {
            throw std::invalid_argument("A scene object cannot parent itself");
        }

        auto ancestor = parentId;
        while (ancestor)
        {
            if (ancestor.value() == childId)
            {
                throw std::invalid_argument("Scene hierarchy cannot contain a cycle");
            }
            ancestor = getSceneObject(ancestor.value()).m_parent;
        }
    }

    if (child.m_parent)
    {
        auto &oldSiblings = getSceneObject(child.m_parent.value()).m_children;
        std::erase(oldSiblings, childId);
    }

    child.m_parent = parentId;
    if (parentId)
    {
        getSceneObject(parentId.value()).m_children.push_back(childId);
    }
}

void Scene::selectObject(SceneObjectId id)
{
    // getSceneObject both validates the ID and rejects retired objects.
    (void)getSceneObject(id);
    m_selectedObject = id;
    m_selectionChangedCallbacks.invoke(id);
}

void Scene::selectHierarchyObject(SceneObjectId id)
{
    // Validate before replacing the current owner so a bad external ID cannot withdraw a valid
    // hierarchy selection. selectObject validates again at its public API boundary.
    (void)getSceneObject(id);
    m_hierarchySelectedObject = id;
    m_hierarchySelectionChangedCallbacks.invoke(id);
    selectObject(id);
}

void Scene::clearHierarchySelection()
{
    m_hierarchySelectedObject.reset();
    m_hierarchySelectionChangedCallbacks.invoke(std::nullopt);
}

void Scene::clearSelection()
{
    m_selectedObject.reset();
    m_hierarchySelectedObject.reset();
    // Both lists hear about it: the hierarchy's owns the Delete target, and the tool-facing one owns
    // the Inspector, the transform gizmo and the selection outline.
    m_hierarchySelectionChangedCallbacks.invoke(std::nullopt);
    m_selectionChangedCallbacks.invoke(std::nullopt);
}

bool Scene::isAncestorOfPendingReveal(SceneObjectId candidate) const
{
    if (!m_pendingReveal || !contains(*m_pendingReveal))
    {
        return false;
    }
    // From the target's parent upwards, so the target's own row is left however the user had it.
    for (std::optional<SceneObjectId> step = find(*m_pendingReveal)->parent(); step;)
    {
        if (*step == candidate)
        {
            return true;
        }
        const SceneObject *object = find(*step);
        step                      = object ? object->parent() : std::nullopt;
    }
    return false;
}

void Scene::drawHierarchyNode(SceneObject &object, std::optional<SceneObjectId> &renameRequested,
                              std::optional<SceneObjectId> &deleteRequested)
{
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick |
                               ImGuiTreeNodeFlags_SpanAvailWidth;
    if (object.children().empty())
    {
        flags |= ImGuiTreeNodeFlags_Leaf;
    }
    if (m_hierarchySelectedObject == object.id())
    {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    // An ancestor of a reveal target is forced open before it is drawn, so the recursion below
    // actually reaches the row the panel is being asked to scroll to.
    if (isAncestorOfPendingReveal(object.id()))
    {
        ImGui::SetNextItemOpen(true);
    }

    const std::string identity = toString(object.id());
    const std::string label    = object.name.empty() ? "Scene Object " + identity : object.name;
    // The identity string is the ImGui ID, so a row keeps its expanded state across frames and
    // across a reload of the same scene.
    const bool open = ImGui::TreeNodeEx(identity.c_str(), flags, "%s", label.c_str());
    if (ImGui::IsItemClicked())
    {
        selectHierarchyObject(object.id());
    }
    if (m_pendingReveal == object.id())
    {
        ImGui::SetScrollHereY(0.5f);
    }

    if (ImGui::BeginPopupContextItem())
    {
        if (ImGui::MenuItem("Rename"))
        {
            renameRequested = object.id();
        }
        ImGui::BeginDisabled(!canDestroySceneObject(object.id()));
        if (ImGui::MenuItem("Delete"))
        {
            deleteRequested = object.id();
        }
        ImGui::EndDisabled();
        ImGui::EndPopup();
    }

    if (open)
    {
        for (SceneObjectId childId : object.children())
        {
            drawHierarchyNode(getSceneObject(childId), renameRequested, deleteRequested);
        }
        ImGui::TreePop();
    }
}

void Scene::drawRenamePopup(EditorContext &context)
{
    if (!m_renamingObject)
    {
        return;
    }

    bool closePopup = false;
    if (ImGui::BeginPopupModal("Rename Scene Object", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        if (!contains(*m_renamingObject))
        {
            closePopup = true;
        }
        else
        {
            if (ImGui::IsWindowAppearing())
            {
                ImGui::SetKeyboardFocusHere();
            }

            const bool submitted = inputText("Name", m_renameBuffer,
                                             ImGuiInputTextFlags_EnterReturnsTrue);
            if (submitted || ImGui::Button("Rename"))
            {
                SceneObject &renamed = getSceneObject(*m_renamingObject);
                if (renamed.name != m_renameBuffer)
                {
                    context.commands.executeCommand(std::make_unique<RenameSceneObjectCommand>(
                        *this, renamed.id(), renamed.name, m_renameBuffer));
                }
                closePopup = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape))
            {
                closePopup = true;
            }
        }

        if (closePopup)
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (closePopup)
    {
        m_renamingObject.reset();
        m_renameBuffer.clear();
    }
}

void Scene::onHierarchyGUI(EditorContext &context)
{
    std::optional<SceneObjectId> renameRequested;
    std::optional<SceneObjectId> deleteRequested;
    for (auto &object : m_sceneObjects)
    {
        if (object->m_alive && !object->parent())
        {
            drawHierarchyNode(*object, renameRequested, deleteRequested);
        }
    }

    // Consumed by the walk above, whether or not the target was found: a request that outlived its
    // object (deleted between the click and this draw) must not keep forcing rows open.
    m_pendingReveal.reset();

    if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        !ImGui::IsAnyItemHovered())
    {
        clearHierarchySelection();
    }

    if (renameRequested)
    {
        m_renamingObject = renameRequested;
        m_renameBuffer   = getSceneObject(*renameRequested).name;
        ImGui::OpenPopup("Rename Scene Object");
    }
    drawRenamePopup(context);

    if (deleteRequested)
    {
        destroySceneObject(*deleteRequested);
    }
}

void Scene::onInspectorGUI(EditorContext &context)
{
    if (!m_selectedObject)
    {
        ImGui::TextDisabled("Select a scene object to inspect it.");
        return;
    }

    SceneObject &object = getSceneObject(m_selectedObject.value());
    ImGui::TextUnformatted(object.name.empty() ? "Unnamed Scene Object" : object.name.c_str());
    ImGui::Separator();
    object.onGUI(context);
    drawAddComponentPanel(object, context);
    // Belongs to the Inspector window itself, so it opens over the object's heading and over the
    // space below its components. Each component draws into a child window of its own and keeps its
    // own menu (Component::onGUI), so a right-click there offers that component's copy/paste rather
    // than this.
    drawComponentPasteMenu(object, context);
}

namespace
{
// Whether `object` can be given a component of `type` at all: it must not already have one, and —
// unless the component is the transform itself — it must have a transform. Everything else in the
// engine reads its object's TransformComponent (the light and camera uploaders, the physics bodies,
// Camera::viewMatrix), so a component added to an object without one would throw on the following
// frame rather than misbehave visibly.
bool canTakeComponent(const SceneObject &object, std::type_index type)
{
    if (object.hasComponent(type))
    {
        return false;
    }
    return type == std::type_index(typeid(TransformComponent)) || object.hasComponent<TransformComponent>();
}

// Records a component the Inspector has just added, reading the values back off the component
// itself so that redoing reproduces the one the user actually got. A component that cannot be
// rebuilt from its values — or an add that quietly did nothing, as pasting a mesh whose geometry
// has since left the store does — is left out of the history rather than given an undo that would
// not put things back.
void recordComponentAdd(SceneObject &object, std::type_index type, EditorContext &context)
{
    Component *added = object.componentOfType(type);
    if (!added)
    {
        return;
    }
    ComponentValuesAdder             adder  = added->valuesAdder();
    std::unique_ptr<ComponentValues> values = added->undoValues();
    if (!adder || !values)
    {
        return;
    }
    context.commands.appendCommandWithoutExecuting(std::make_unique<AddComponentCommand>(
        object.scene(), object.id(), type, std::move(values), std::move(adder), context.componentAdds));
}
} // namespace

void Scene::drawComponentPasteMenu(SceneObject &object, EditorContext &context)
{
    if (!ImGui::BeginPopupContextWindow())
    {
        return;
    }

    const ComponentClipboard            &clipboard = context.componentClipboard;
    const std::optional<std::type_index> copied    = clipboard.componentType();
    const bool addable = copied && clipboard.values() && clipboard.adder();

    ImGui::BeginDisabled(!addable || !canTakeComponent(object, *copied));
    if (ImGui::MenuItem("Paste component"))
    {
        clipboard.adder()(object, *clipboard.values());
        context.componentAdds.onComponentAdded(object);
        recordComponentAdd(object, *copied, context);
    }
    ImGui::EndDisabled();
    ImGui::EndPopup();
}

void Scene::drawAddComponentPanel(SceneObject &object, EditorContext &context)
{
    // A window of its own below the components, so the Inspector reads as the object's components
    // followed by the way to add another.
    const bool visible = ImGui::BeginChild("add_component", ImVec2(0.0f, 0.0f),
                                           ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    if (visible)
    {
        if (ImGui::Button("Add Component"))
        {
            ImGui::OpenPopup("add_component_list");
        }
        if (ImGui::BeginPopup("add_component_list"))
        {
            // Every type the editor offers is listed, with the ones this object cannot take disabled
            // rather than missing, so the list reads the same way on every object.
            for (const ComponentCatalog::Entry &entry : context.componentCatalog.entries())
            {
                ImGui::BeginDisabled(!canTakeComponent(object, entry.type));
                if (ImGui::MenuItem(entry.name.c_str()))
                {
                    entry.add(object);
                    context.componentAdds.onComponentAdded(object);
                    recordComponentAdd(object, entry.type, context);
                }
                ImGui::EndDisabled();
            }
            ImGui::EndPopup();
        }
    }
    ImGui::EndChild();
}

} // namespace lr
