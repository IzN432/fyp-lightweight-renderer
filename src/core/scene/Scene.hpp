#pragma once

#include "SceneObject.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace lr
{

class Scene
{
public:
    Scene() = default;

    void onHierarchyGUI();
    void onInspectorGUI();

    SceneObject &createSceneObject();

    SceneObject       &getSceneObject(SceneObjectId id);
    const SceneObject &getSceneObject(SceneObjectId id) const;

    void setParent(SceneObjectId child, std::optional<SceneObjectId> parent);

    const std::vector<std::unique_ptr<SceneObject>> &sceneObjects() const { return m_sceneObjects; }

private:
    void drawHierarchyNode(SceneObject &object);

    std::vector<std::unique_ptr<SceneObject>> m_sceneObjects;
    std::optional<SceneObjectId>               m_selectedObject;
};

} // namespace lr
