#pragma once

#include <memory>

namespace lr
{

struct EditorContext;
class EditorRenderBridge;
class ErrorReporter;
class SceneDocument;
class SceneManager;
class SceneObject;
class Viewer;

// Application-level editor facade. It owns the interaction objects that must move in lockstep
// when the hierarchy selection or edited mesh changes. Keeping this separate from SceneManager
// avoids making scene/GPU synchronization depend on optional editor features; Engine owns both
// objects without exposing this orchestration through the executable entry point.
class EditorSession
{
public:
    EditorSession(Viewer &viewer, SceneManager &sceneManager, SceneObject &camera,
                  EditorRenderBridge &renderBridge, ErrorReporter &errors);
    ~EditorSession();

    EditorSession(const EditorSession &)            = delete;
    EditorSession &operator=(const EditorSession &) = delete;

    EditorContext &context();

    // Rebinds feature state and render sources after a whole-scene load or another operation that
    // changes SceneManager's edited mesh without going through a hierarchy-selection callback.
    void onSceneContentChanged();

    void drawFeaturePanel();

    // The Scene window's contents: Hierarchy (document errors, the import/create buttons, the object
    // tree) above Inspector. The caller owns the window itself — where it sits and how big it is are
    // the host application's layout decisions, not the editor's. `document` supplies the error text
    // and receives the Import button's dialog request; the caller resolves that dialog by calling
    // SceneDocument::drawDialogs() outside any window.
    void drawScenePanel(SceneDocument &document);
    void drawTransformWindow();
    void drawAnimationWindow();
    void drawSkinningWindow();

    // Whether camera orbit/pan/zoom may act on the pointer this frame — the editor arbitrates
    // pointer priority, so a host camera controller asks rather than guessing.
    bool allowsViewportNavigation() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace lr
