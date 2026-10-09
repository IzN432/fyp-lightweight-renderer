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

    // Retires an object and its complete subtree. IDs and object addresses are not reused during
    // ordinary editing, so stale editor references cannot bind to a different object. Destruction
    // callbacks run after retirement so observers see the new scene state. A full-scene replacement
    // may subsequently release the tombstones with purgeDestroyedSceneObjects().
    void destroySceneObject(SceneObjectId id);
    bool contains(SceneObjectId id) const;
    bool canDestroySceneObject(SceneObjectId id) const;

    // Permanently removes retired object storage and releases its IDs. Only use at a full-scene
    // replacement boundary, after every observer has processed the destruction callbacks.
    void purgeDestroyedSceneObjects();

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

    // The object currently presented to scene-facing editor tools. Other surfaces (for example an
    // animation track) may drive this without claiming ownership of destructive actions.
    std::optional<SceneObjectId> selectedObject() const { return m_selectedObject; }

    // The resource selected by the hierarchy/viewport. Kept separate from selectedObject() so an
    // animation track can present its target in the scene without making Delete destroy it.
    std::optional<SceneObjectId> hierarchySelectedObject() const { return m_hierarchySelectedObject; }
    void selectHierarchyObject(SceneObjectId id);
    void clearHierarchySelection();

    // Ask the hierarchy panel to expand every ancestor of `id` and scroll the row into view on its
    // next draw. For selections made somewhere else — a viewport click — whose object may sit inside
    // collapsed parents; a click in the panel is already looking at the row.
    void revealInHierarchy(SceneObjectId id) { m_pendingReveal = id; }

    CallbackConnection registerHierarchySelectionChangedCallback(
        std::function<void(std::optional<SceneObjectId>)> callback)
    {
        return m_hierarchySelectionChangedCallbacks.connect(std::move(callback));
    }

    // Routes every selection source through the same validation and notification path.
    void selectObject(SceneObjectId id);

    // Nothing selected, in both senses: the object offered to tools and the hierarchy's own row.
    // A viewport click on empty space lands here. (The panel's own empty-space click withdraws only
    // its row, through clearHierarchySelection, so a target another surface is presenting stays put.)
    void clearSelection();

    // Fired whenever selectObject() is called, even if the object was already selected, and with
    // nullopt when the selection is cleared.
    CallbackConnection registerSelectionChangedCallback(std::function<void(std::optional<SceneObjectId>)> callback)
    {
        return m_selectionChangedCallbacks.connect(std::move(callback));
    }

    CallbackConnection registerObjectsDestroyedCallback(
        std::function<void(std::span<const SceneObjectId>)> callback)
    {
        return m_objectsDestroyedCallbacks.connect(std::move(callback));
    }

private:
    void drawHierarchyNode(SceneObject &object, std::optional<SceneObjectId> &renameRequested,
                           std::optional<SceneObjectId> &deleteRequested);
    // True while a reveal is pending and `candidate` is a proper ancestor of it. The target itself
    // is excluded: revealing a row means getting to it, not expanding its own children.
    bool isAncestorOfPendingReveal(SceneObjectId candidate) const;
    void drawRenamePopup();
    // The Inspector's own context menu: adds the component on the clipboard to `object`.
    void drawComponentPasteMenu(SceneObject &object, EditorContext &context);
    // The panel below `object`'s components, offering the editor's component catalog.
    void drawAddComponentPanel(SceneObject &object, EditorContext &context);

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
    std::optional<SceneObjectId>               m_hierarchySelectedObject;
    std::optional<SceneObjectId>               m_pendingReveal;
    std::optional<SceneObjectId>               m_renamingObject;
    std::string                                m_renameBuffer;
    CallbackList<std::optional<SceneObjectId>>      m_selectionChangedCallbacks;
    CallbackList<std::optional<SceneObjectId>>      m_hierarchySelectionChangedCallbacks;
    CallbackList<std::span<const SceneObjectId>>    m_objectsDestroyedCallbacks;
    std::unordered_set<SceneObjectId> m_protectedObjects;
};

} // namespace lr
