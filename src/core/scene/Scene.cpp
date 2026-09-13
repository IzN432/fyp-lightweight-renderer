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
    if (id >= m_sceneObjects.size())
    {
        throw std::out_of_range("Scene object ID is out of range");
    }
    return *m_sceneObjects[id];
}

const SceneObject &Scene::getSceneObject(SceneObjectId id) const
{
    if (id >= m_sceneObjects.size())
    {
        throw std::out_of_range("Scene object ID is out of range");
    }
    return *m_sceneObjects[id];
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

void Scene::drawHierarchyNode(SceneObject &object)
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
    }

    if (open)
    {
        for (SceneObjectId childId : object.children())
        {
            drawHierarchyNode(getSceneObject(childId));
        }
        ImGui::TreePop();
    }
}

void Scene::onHierarchyGUI()
{
    for (auto &object : m_sceneObjects)
    {
        if (!object->parent())
        {
            drawHierarchyNode(*object);
        }
    }
}

void Scene::onInspectorGUI()
{
    if (!m_selectedObject)
    {
        ImGui::TextDisabled("Select a scene object to inspect it.");
        return;
    }

    SceneObject &object = getSceneObject(m_selectedObject.value());
    ImGui::TextUnformatted(object.name.empty() ? "Unnamed Scene Object" : object.name.c_str());
    ImGui::Separator();
    object.onGUI();
}

} // namespace lr
