#pragma once

#include "core/editor/EditorPresentation.hpp"

namespace lr
{

class AreaShadowPass;
class CascadedShadowPass;
class GeometryPass;
class HeatmapPass;
class ObjectPickingPass;
class OverlayLinesPass;
class OverlayPointsPass;
class Scene;
class SceneManager;
class SpotShadowPass;
class TransparentPass;
struct EditableMeshContext;
struct EditorFrameContext;

// The only application-editor object that knows concrete render passes. EditorSession publishes
// presentation, target, and per-frame overlay data through this boundary instead of coordinating
// pass visibility and resource rebinding itself.
class EditorRenderBridge
{
public:
    // The passes the bridge drives. A struct rather than ten positional parameters, so the call
    // site names each one.
    struct Passes
    {
        GeometryPass       *geometry       = nullptr;
        TransparentPass    *transparent    = nullptr;
        ObjectPickingPass  *objectPicking  = nullptr;
        SpotShadowPass     *spotShadow     = nullptr;
        CascadedShadowPass *cascadedShadow = nullptr;
        AreaShadowPass     *areaShadow     = nullptr;
        HeatmapPass        *heatmap        = nullptr;
        OverlayPointsPass  *overlayPoints  = nullptr;
        OverlayLinesPass   *overlayLines   = nullptr;
    };

    // Every member of `passes` must be non-null.
    EditorRenderBridge(SceneManager &sceneManager, Passes passes);

    void apply(const EditorPresentation &presentation);
    void setEditableTarget(const EditableMeshContext &target);
    void updateSceneOverlays(const Scene &scene, const EditorFrameContext &frame);

private:
    SceneManager &m_sceneManager;
    Passes        m_passes;
};

} // namespace lr
