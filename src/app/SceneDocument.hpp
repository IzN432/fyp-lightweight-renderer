#pragma once

#include "core/loaders/ObjExporter.hpp"
#include "core/loaders/SceneLoader.hpp"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace lr
{

class EditorSession;
class ErrorReporter;
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
                  SceneLoaderConfig importConfig, ErrorReporter &errors);

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

    // Exports every renderable object in the scene as a single Wavefront OBJ: the registered mesh
    // objects the renderer is drawing, each as its own `o` group with its positions baked through
    // its world transform, so the file reproduces the scene as composed rather than stacking every
    // mesh on the origin. Geometry only -- positions and faces, no normals, uvs or materials (see
    // ObjExporter). Disabled when the scene has no renderable geometry, which canExportGeometry()
    // reports for the menu item.
    void openExportGeometryDialog();

    // Whether the scene has any renderable mesh geometry to export.
    bool canExportGeometry();

    // Imports one of the built-in primitives shipped in assets/runtime/primitives/, naming its root
    // after the shape rather than after the file. Takes the same path as a user import — including
    // the device idle and the resettle afterwards — so a primitive is an ordinary imported object
    // from then on, editable and deletable like any other. Failures land in importError() as an
    // import does.
    void importPrimitive(const std::string &fileName, const std::string &objectName);

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

    // The scene's renderable mesh objects as exporter entries, each paired with its world transform.
    // Taken from the registered mesh objects, which are what the renderer draws, so the export
    // matches what is on screen.
    std::vector<ObjExportObject> collectExportObjects() const;

    Viewer           &m_viewer;
    SceneManager     &m_sceneManager;
    EditorSession    &m_editor;
    PhysicsWorld     &m_physicsWorld;
    ErrorReporter    &m_errors;
    SceneLoaderConfig m_importConfig;

    std::function<void()> m_onSceneLoaded;

    std::optional<std::filesystem::path> m_path;
    std::string                          m_persistenceError;
    std::string                          m_importError;
};

} // namespace lr
