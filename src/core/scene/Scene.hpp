#pragma once

#include "SceneObject.hpp"
#include "core/utility/CallbackList.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
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

    // Recreates an object under an identity it already had, for loading a saved scene. Throws if
    // the id is nil or already belongs to an object in this scene, so a collision is loud rather
    // than silently rebinding a live reference.
    SceneObject &createSceneObject(SceneObjectId id);

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
    CallbackConnection registerSelectionChangedCallback(std::function<void(SceneObjectId)> callback)
    {
        return m_selectionChangedCallbacks.connect(std::move(callback));
    }

    CallbackConnection registerObjectsDestroyedCallback(
        std::function<void(std::span<const SceneObjectId>)> callback)
    {
        return m_objectsDestroyedCallbacks.connect(std::move(callback));
    }

private:
    void drawHierarchyNode(SceneObject &object, std::optional<SceneObjectId> &deleteRequested);

    // Finds an object whether or not it is still alive — destroySceneObject() has to reach objects
    // it is in the middle of retiring. Callers that need a live object go through getSceneObject().
    SceneObject       *find(SceneObjectId id);
    const SceneObject *find(SceneObjectId id) const;

    // Objects are stored in creation order and looked up by identity. The vector is what
    // sceneObjects() hands out, and iterating it keeps scene traversal and saved-file ordering
    // deterministic, which a hash map's iteration order would not.
    std::vector<std::unique_ptr<SceneObject>> m_sceneObjects;
    std::unordered_map<SceneObjectId, size_t> m_objectIndices;
    std::optional<std::filesystem::path>       m_hdriPath;
    std::vector<std::byte>                     m_hdriData;
    std::optional<SceneObjectId>               m_selectedObject;
    CallbackList<SceneObjectId>                     m_selectionChangedCallbacks;
    CallbackList<std::span<const SceneObjectId>>    m_objectsDestroyedCallbacks;
    std::unordered_set<SceneObjectId> m_protectedObjects;
};

} // namespace lr
