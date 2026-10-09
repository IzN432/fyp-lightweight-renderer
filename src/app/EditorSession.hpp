#pragma once

#include <memory>

namespace lr
{

struct EditorContext;
class EditorRenderBridge;
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
                  EditorRenderBridge &renderBridge);
    ~EditorSession();

    EditorSession(const EditorSession &)            = delete;
    EditorSession &operator=(const EditorSession &) = delete;

    EditorContext &context();

    // Rebinds feature state and render sources after a whole-scene load or another operation that
    // changes SceneManager's edited mesh without going through a hierarchy-selection callback.
    void onSceneContentChanged();

    void drawFeaturePanel();
    void drawTransformWindow();
    void drawAnimationWindow();

    // Whether camera orbit/pan/zoom may act on the pointer this frame — the editor arbitrates
    // pointer priority, so a host camera controller asks rather than guessing.
    bool allowsViewportNavigation() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace lr
