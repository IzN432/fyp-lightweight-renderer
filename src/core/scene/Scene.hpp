#pragma once

#include "SceneObject.hpp"

#include <functional>
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

    // The object currently selected in the Scene Hierarchy panel, if any — nullopt until the user
    // has clicked a row.
    std::optional<SceneObjectId> selectedObject() const { return m_selectedObject; }

    // Fired whenever a Scene Hierarchy row is clicked, with the newly selected object's id — even
    // if it's the object that was already selected (callers that only care about actual changes
    // should compare against their own last-seen id).
    void registerSelectionChangedCallback(std::function<void(SceneObjectId)> callback)
    {
        m_selectionChangedCallbacks.push_back(std::move(callback));
    }

private:
    void drawHierarchyNode(SceneObject &object);

    std::vector<std::unique_ptr<SceneObject>> m_sceneObjects;
    std::optional<SceneObjectId>               m_selectedObject;
    std::vector<std::function<void(SceneObjectId)>> m_selectionChangedCallbacks;
};

} // namespace lr
