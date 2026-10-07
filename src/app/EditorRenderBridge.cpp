#include "EditorRenderBridge.hpp"

#include "core/editor/EditableMeshContext.hpp"
#include "core/editor/EditorFrameContext.hpp"
#include "core/passes/geometry/GeometryPass.hpp"
#include "core/passes/heatmap/HeatmapPass.hpp"
#include "core/passes/overlaylines/OverlayLinesPass.hpp"
#include "core/passes/overlaypoints/OverlayPointsPass.hpp"
#include "core/passes/transparent/TransparentPass.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneManager.hpp"
#include "core/scene/TransformComponent.hpp"
#include "features/rigid_body/ColliderVisual.hpp"

#include <optional>
#include <vector>

namespace lr
{

EditorRenderBridge::EditorRenderBridge(SceneManager &sceneManager, GeometryPass &geometryPass,
                                       TransparentPass &transparentPass, HeatmapPass &heatmapPass,
                                       OverlayPointsPass &overlayPointsPass, OverlayLinesPass &overlayLinesPass)
    : m_sceneManager(sceneManager), m_geometryPass(geometryPass), m_transparentPass(transparentPass),
      m_heatmapPass(heatmapPass), m_overlayPointsPass(overlayPointsPass),
      m_overlayLinesPass(overlayLinesPass)
{}

void EditorRenderBridge::apply(const EditorPresentation &presentation)
{
    m_geometryPass.setSkinningEnabled(presentation.skinningEnabled);
    m_transparentPass.setSkinningEnabled(presentation.skinningEnabled);
    m_overlayPointsPass.setEnabled(presentation.vertexPointsVisible);
    m_heatmapPass.setEnabled(presentation.heatmapVisible);
}

void EditorRenderBridge::setEditableTarget(const EditableMeshContext &target)
{
    auto &transform = target.object.getComponent<TransformComponent>();
    m_heatmapPass.setMeshSource(m_sceneManager.selectedMeshHeatmap(),
                                m_sceneManager.selectedMeshIndexRange(), transform);
    m_overlayPointsPass.setPointsSource(m_sceneManager.selectedMeshPoints(),
                                        target.mesh.uniquePositionCount(), transform);
}

void EditorRenderBridge::updateSceneOverlays(const Scene &scene, const EditorFrameContext &frame)
{
    std::vector<OverlayLine> overlayLines = buildColliderOverlayLines(scene);
    if (const std::optional<SceneObjectId> selected = scene.selectedObject())
    {
        OverlayLineBuilder selectionGizmo;
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
    m_overlayLinesPass.setLines(overlayLines);
}

} // namespace lr
