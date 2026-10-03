#include "Scene.hpp"

#include <imgui.h>

#include <algorithm>
#include <stdexcept>

namespace lr
{

SceneObject &Scene::createSceneObject()
{
    const SceneObjectId id = static_cast<SceneObjectId>(m_sceneObjects.size());
    m_sceneObjects.push_back(std::unique_ptr<SceneObject>(new SceneObject(*this, id)));
    return *m_sceneObjects.back();
}

SceneObject &Scene::getSceneObject(SceneObjectId id)
{
    if (!contains(id))
    {
        throw std::out_of_range("Scene object ID is out of range");
    }
    return *m_sceneObjects[id];
}

const SceneObject &Scene::getSceneObject(SceneObjectId id) const
{
    if (!contains(id))
    {
        throw std::out_of_range("Scene object ID is out of range");
    }
    return *m_sceneObjects[id];
}

bool Scene::contains(SceneObjectId id) const
{
    return id < m_sceneObjects.size() && m_sceneObjects[id] && m_sceneObjects[id]->m_alive;
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
        const auto &children = m_sceneObjects[current]->m_children;
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
        const auto &children = m_sceneObjects[current]->m_children;
        pending.insert(pending.end(), children.begin(), children.end());
    }

    SceneObject &root = *m_sceneObjects[id];
    if (root.m_parent && contains(*root.m_parent))
    {
        std::erase(m_sceneObjects[*root.m_parent]->m_children, id);
    }
    for (SceneObjectId destroyedId : destroyed)
    {
        SceneObject &object = *m_sceneObjects[destroyedId];
        object.m_parent.reset();
        object.m_children.clear();
        object.m_alive = false;
    }
    if (m_selectedObject && std::ranges::find(destroyed, *m_selectedObject) != destroyed.end())
    {
        m_selectedObject.reset();
    }
    for (const auto &callback : m_objectsDestroyedCallbacks)
    {
        callback(destroyed);
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

void Scene::drawHierarchyNode(SceneObject &object, std::optional<SceneObjectId> &deleteRequested)
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

    const std::string label = object.name.empty() ? "Scene Object " + std::to_string(object.id()) : object.name;
    const bool open = ImGui::TreeNodeEx(reinterpret_cast<void *>(static_cast<uintptr_t>(object.id()) + 1), flags,
                                        "%s", label.c_str());
    if (ImGui::IsItemClicked())
    {
        m_selectedObject = object.id();
        for (const auto &callback : m_selectionChangedCallbacks)
        {
            callback(object.id());
        }
    }

    if (ImGui::BeginPopupContextItem())
    {
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
            drawHierarchyNode(getSceneObject(childId), deleteRequested);
        }
        ImGui::TreePop();
    }
}

void Scene::onHierarchyGUI()
{
    std::optional<SceneObjectId> deleteRequested;
    for (auto &object : m_sceneObjects)
    {
        if (object->m_alive && !object->parent())
        {
            drawHierarchyNode(*object, deleteRequested);
        }
    }

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
