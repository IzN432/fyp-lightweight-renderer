#pragma once

#include "core/editor/EditorPresentation.hpp"

namespace lr
{

class GeometryPass;
class HeatmapPass;
class OverlayLinesPass;
class OverlayPointsPass;
class Scene;
class SceneManager;
class TransparentPass;
struct EditableMeshContext;
struct EditorFrameContext;

// The only application-editor object that knows concrete render passes. EditorSession publishes
// presentation, target, and per-frame overlay data through this boundary instead of coordinating
// pass visibility and resource rebinding itself.
class EditorRenderBridge
{
public:
    EditorRenderBridge(SceneManager &sceneManager, GeometryPass &geometryPass,
                       TransparentPass &transparentPass, HeatmapPass &heatmapPass,
                       OverlayPointsPass &overlayPointsPass, OverlayLinesPass &overlayLinesPass);

    void apply(const EditorPresentation &presentation);
    void setEditableTarget(const EditableMeshContext &target);
    void updateSceneOverlays(const Scene &scene, const EditorFrameContext &frame);

private:
    SceneManager       &m_sceneManager;
    GeometryPass       &m_geometryPass;
    TransparentPass    &m_transparentPass;
    HeatmapPass        &m_heatmapPass;
    OverlayPointsPass  &m_overlayPointsPass;
    OverlayLinesPass   &m_overlayLinesPass;
};

} // namespace lr
