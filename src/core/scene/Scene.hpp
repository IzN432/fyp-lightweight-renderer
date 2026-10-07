#pragma once

#include "SceneObject.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

namespace lr
{

class Scene
{
public:
    Scene() = default;

    void onHierarchyGUI();
    void onInspectorGUI(EditorContext &context);

    SceneObject &createSceneObject();

    // Retires an object and its complete subtree. IDs and object addresses are never reused, so
    // editor commands that still mention an old object cannot become references to a different
    // object. Destruction callbacks run after retirement so observers see the new scene state.
    void destroySceneObject(SceneObjectId id);
    bool contains(SceneObjectId id) const;
    bool canDestroySceneObject(SceneObjectId id) const;

    void protectSceneObject(SceneObjectId id) { m_protectedObjects.insert(id); }

    SceneObject       &getSceneObject(SceneObjectId id);
    const SceneObject &getSceneObject(SceneObjectId id) const;

    void setParent(SceneObjectId child, std::optional<SceneObjectId> parent);

    const std::vector<std::unique_ptr<SceneObject>> &sceneObjects() const { return m_sceneObjects; }

    // Optional environment map belonging to the authored scene. Rendering code decides how to
    // turn it into IBL resources; keeping the source path here lets native scenes persist it.
    const std::optional<std::filesystem::path> &hdriPath() const { return m_hdriPath; }
    void setHdriPath(std::optional<std::filesystem::path> path)
    {
        m_hdriPath = std::move(path);
        m_hdriData.clear();
    }
    const std::vector<std::byte> &hdriData() const { return m_hdriData; }
    void setEmbeddedHdri(std::filesystem::path name, std::vector<std::byte> data)
    {
        m_hdriPath = std::move(name);
        m_hdriData = std::move(data);
    }

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

    void registerObjectsDestroyedCallback(std::function<void(std::span<const SceneObjectId>)> callback)
    {
        m_objectsDestroyedCallbacks.push_back(std::move(callback));
    }

private:
    void drawHierarchyNode(SceneObject &object, std::optional<SceneObjectId> &deleteRequested);

    std::vector<std::unique_ptr<SceneObject>> m_sceneObjects;
    std::optional<std::filesystem::path>       m_hdriPath;
    std::vector<std::byte>                     m_hdriData;
    std::optional<SceneObjectId>               m_selectedObject;
    std::vector<std::function<void(SceneObjectId)>> m_selectionChangedCallbacks;
    std::vector<std::function<void(std::span<const SceneObjectId>)>> m_objectsDestroyedCallbacks;
    std::unordered_set<SceneObjectId> m_protectedObjects;
};

} // namespace lr
