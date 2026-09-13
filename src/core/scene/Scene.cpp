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

void Scene::onGUI()
{
    ImGui::SeparatorText("Scene Objects");

    int id = 0;
    for (auto &object : m_sceneObjects)
    {
        ImGui::PushID(id++);
        object->onGUI();
        ImGui::PopID();
    }
}

} // namespace lr
