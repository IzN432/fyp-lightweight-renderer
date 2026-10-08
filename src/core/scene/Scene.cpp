#include "Scene.hpp"

#include <imgui.h>

#include <algorithm>
#include <stdexcept>

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

void Scene::drawHierarchyNode(SceneObject &object, std::optional<SceneObjectId> &renameRequested,
                              std::optional<SceneObjectId> &deleteRequested)
{
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick |
                               ImGuiTreeNodeFlags_SpanAvailWidth;
    if (object.children().empty())
    {
        flags |= ImGuiTreeNodeFlags_Leaf;
    }
    if (m_selectedObject == object.id())
    {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    const std::string identity = toString(object.id());
    const std::string label    = object.name.empty() ? "Scene Object " + identity : object.name;
    // The identity string is the ImGui ID, so a row keeps its expanded state across frames and
    // across a reload of the same scene.
    const bool open = ImGui::TreeNodeEx(identity.c_str(), flags, "%s", label.c_str());
    if (ImGui::IsItemClicked())
    {
        selectObject(object.id());
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

void Scene::drawRenamePopup()
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
                getSceneObject(*m_renamingObject).name = m_renameBuffer;
                closePopup                            = true;
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

void Scene::onHierarchyGUI()
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

    if (renameRequested)
    {
        m_renamingObject = renameRequested;
        m_renameBuffer   = getSceneObject(*renameRequested).name;
        ImGui::OpenPopup("Rename Scene Object");
    }
    drawRenamePopup();

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
}

} // namespace lr
