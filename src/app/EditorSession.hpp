#pragma once

#include <memory>

namespace lr
{

struct EditorContext;
class GeometryPass;
class HeatmapPass;
class OverlayLinesPass;
class OverlayPointsPass;
class SceneManager;
class SceneObject;
class TransparentPass;
class Viewer;

// Application-level editor facade. It owns the interaction objects that must move in lockstep
// when the hierarchy selection or edited mesh changes. Keeping this separate from SceneManager
// avoids making scene/GPU synchronization depend on optional editor features; a future Engine can
// own both objects without moving this orchestration back into main.cpp.
class EditorSession
{
public:
    EditorSession(Viewer &viewer, SceneManager &sceneManager, SceneObject &camera,
                  GeometryPass &geometryPass, TransparentPass &transparentPass,
                  HeatmapPass &heatmapPass, OverlayPointsPass &overlayPointsPass,
                  OverlayLinesPass &overlayLinesPass);
    ~EditorSession();

    EditorSession(const EditorSession &)            = delete;
    EditorSession &operator=(const EditorSession &) = delete;

    EditorContext &context();

    // Rebinds feature state and render sources after a whole-scene load or another operation that
    // changes SceneManager's edited mesh without going through a hierarchy-selection callback.
    void onSceneContentChanged();

    void drawFeaturePanel();
    void drawTransformWindow();

    // Returns if the editor gizmos are currently capturing the mouse
    bool capturesMouse() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace lr
