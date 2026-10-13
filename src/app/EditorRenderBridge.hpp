#pragma once

#include "core/editor/EditorPresentation.hpp"
#include "core/passes/overlaygeometry/OverlayInstance.hpp"

#include <functional>
#include <vector>

namespace lr
{

class AreaShadowPass;
class CascadedShadowPass;
class GeometryPass;
class HeatmapPass;
class ObjectPickingPass;
class OverlayGeometryPass;
class OverlayLineBuilder;
class OverlayLinesPass;
class OutlinePass;
class OverlayPointsPass;
class Scene;
class SceneManager;
class SpotShadowPass;
class PointShadowPass;
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
        PointShadowPass    *pointShadow    = nullptr;
        CascadedShadowPass *cascadedShadow = nullptr;
        AreaShadowPass     *areaShadow     = nullptr;
        HeatmapPass        *heatmap        = nullptr;
        OverlayGeometryPass *overlayGeometry = nullptr;
        OverlayPointsPass  *overlayPoints  = nullptr;
        OverlayLinesPass   *overlayLines   = nullptr;
        OutlinePass        *outline        = nullptr;
    };

    // Every member of `passes` must be non-null.
    EditorRenderBridge(SceneManager &sceneManager, Passes passes);

    void apply(const EditorPresentation &presentation);
    void setEditableTarget(const EditableMeshContext &target);
    // Registers a producer of overlay line segments that does not come from the scene graph, such
    // as a simulation's bounding box. The bridge owns the single setLines() call, so anything that
    // wants lines on screen contributes through here rather than driving the pass itself.
    void addOverlayLineProducer(std::function<void(OverlayLineBuilder &)> producer);
    void setOverlayGeometry(std::vector<OverlayInstance> instances);
    void setHoveredOverlay(uint32_t pickingId);
    void updateSceneOverlays(const Scene &scene, const EditorFrameContext &frame);

private:
    SceneManager &m_sceneManager;
    Passes        m_passes;
    std::vector<std::function<void(OverlayLineBuilder &)>> m_lineProducers;
};

} // namespace lr
