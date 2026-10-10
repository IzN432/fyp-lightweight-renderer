#pragma once

#include "core/loaders/SceneLoader.hpp"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>

namespace lr
{

class EditorSession;
class PhysicsWorld;
class SceneManager;
class Viewer;

// Load/save/import of the authored scene document, and the file dialogs that drive them.
//
// Each of these operations is a fixed sequence across four subsystems — idle the device, mutate the
// scene, re-pack geometry, rebind the editor's edited-mesh state, resettle physics, recompile the
// frame graph — and getting that order wrong produces use-after-free of GPU resources rather than a
// visible mistake. Collecting it here keeps the sequence in one place and off the GUI callback,
// which previously spelled it out once per dialog.
//
// What a loaded scene means for rendering policy stays with the application: onSceneLoaded() fires
// after a successful whole-scene load so the host can rebuild anything derived from the document
// (the IBL environment, for one) without this class knowing those passes exist.
class SceneDocument
{
public:
    SceneDocument(Viewer &viewer, SceneManager &sceneManager, EditorSession &editor, PhysicsWorld &physicsWorld,
                  SceneLoaderConfig importConfig);

    // Fires after loadScene() succeeds, for host-owned state derived from the document.
    void setSceneLoadedCallback(std::function<void()> callback) { m_onSceneLoaded = std::move(callback); }

    // The file backing "Save". Empty until a scene is loaded or saved to a path.
    const std::optional<std::filesystem::path> &path() const { return m_path; }

    // Last load/save failure, and last import failure. Kept apart because the panel reports them in
    // different places — a document error belongs with the document, an import error with the
    // Import button. Cleared by the next attempt that succeeds.
    const std::string &persistenceError() const { return m_persistenceError; }
    const std::string &importError() const { return m_importError; }

    // Opens the corresponding modal. The dialog itself resolves in drawDialogs().
    void openLoadDialog();
    void openSaveAsDialog();
    void openImportDialog();

    // Saves to path() without prompting. Only valid when path() has a value; the menu item that
    // calls this is disabled otherwise.
    void save();

    // Runs the three file dialogs this class opened, applying whichever one the user confirmed.
    // Call once per frame, outside any ImGui window — a modal cannot open from inside one.
    void drawDialogs();

private:
    // Shared tail of load/import: both replace geometry the GPU may still be reading, so both idle
    // the device first (done by the caller, before mutating the scene) and rebuild everything bound
    // against the old contents afterward.
    void resettleAfterSceneChange();

    Viewer           &m_viewer;
    SceneManager     &m_sceneManager;
    EditorSession    &m_editor;
    PhysicsWorld     &m_physicsWorld;
    SceneLoaderConfig m_importConfig;

    std::function<void()> m_onSceneLoaded;

    std::optional<std::filesystem::path> m_path;
    std::string                          m_persistenceError;
    std::string                          m_importError;
};

} // namespace lr
