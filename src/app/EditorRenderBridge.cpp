#include "EditorRenderBridge.hpp"

#include "core/editor/EditableMeshContext.hpp"
#include "core/editor/EditorFrameContext.hpp"
#include "core/passes/geometry/GeometryPass.hpp"
#include "core/passes/heatmap/HeatmapPass.hpp"
#include "core/passes/objectpicking/ObjectPickingPass.hpp"
#include "core/passes/overlaylines/OverlayLinesPass.hpp"
#include "core/passes/overlaypoints/OverlayPointsPass.hpp"
#include "core/passes/shadow/AreaShadowPass.hpp"
#include "core/passes/shadow/CascadedShadowPass.hpp"
#include "core/passes/shadow/SpotShadowPass.hpp"
#include "core/passes/transparent/TransparentPass.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneManager.hpp"
#include "core/scene/TransformComponent.hpp"
#include "features/rigid_body/ColliderVisual.hpp"

#include <optional>
#include <stdexcept>
#include <vector>

namespace lr
{

EditorRenderBridge::EditorRenderBridge(SceneManager &sceneManager, Passes passes)
    : m_sceneManager(sceneManager), m_passes(passes)
{
    if (!m_passes.geometry || !m_passes.transparent || !m_passes.objectPicking || !m_passes.spotShadow ||
        !m_passes.cascadedShadow || !m_passes.areaShadow || !m_passes.heatmap || !m_passes.overlayPoints ||
        !m_passes.overlayLines)
    {
        throw std::invalid_argument("EditorRenderBridge: every pass is required");
    }
}

void EditorRenderBridge::apply(const EditorPresentation &presentation)
{
    // Every pass that replays scene geometry follows the same toggle. A mesh-editing state draws the
    // rest pose so the vertex overlays and gizmos line up with it, and shadows, picking IDs and the
    // transparent layer have to agree with what the G-buffer drew or they describe a pose that is
    // not on screen.
    m_passes.geometry->setSkinningEnabled(presentation.skinningEnabled);
    m_passes.transparent->setSkinningEnabled(presentation.skinningEnabled);
    m_passes.objectPicking->setSkinningEnabled(presentation.skinningEnabled);
    m_passes.spotShadow->setSkinningEnabled(presentation.skinningEnabled);
    m_passes.cascadedShadow->setSkinningEnabled(presentation.skinningEnabled);
    m_passes.areaShadow->setSkinningEnabled(presentation.skinningEnabled);

    m_passes.overlayPoints->setEnabled(presentation.vertexPointsVisible);
    m_passes.heatmap->setEnabled(presentation.heatmapVisible);
}

void EditorRenderBridge::setEditableTarget(const EditableMeshContext &target)
{
    auto &transform = target.object.getComponent<TransformComponent>();
    m_passes.heatmap->setMeshSource(m_sceneManager.selectedMeshHeatmap(), m_sceneManager.selectedMeshIndexRange(),
                                    transform);
    m_passes.overlayPoints->setPointsSource(m_sceneManager.selectedMeshPoints(), target.mesh.uniquePositionCount(),
                                            transform);
}

void EditorRenderBridge::updateSceneOverlays(const Scene &scene, const EditorFrameContext &frame)
{
    std::vector<OverlayLine> overlayLines = buildColliderOverlayLines(scene);
    if (const std::optional<SceneObjectId> selected = scene.selectedObject())
    {
        OverlayLineBuilder    selectionGizmo;
        SelectionGizmoContext context{
            .lines          = selectionGizmo,
            .cameraPosition = frame.cameraPosition,
            .cameraForward  = frame.cameraForward,
            .orthographic   = frame.orthographic,
        };
        scene.getSceneObject(*selected).onSelectGizmo(context);
        std::vector<OverlayLine> selectionLines = selectionGizmo.takeLines();
        overlayLines.insert(overlayLines.end(), selectionLines.begin(), selectionLines.end());
    }
    m_passes.overlayLines->setLines(overlayLines);
}

} // namespace lr
