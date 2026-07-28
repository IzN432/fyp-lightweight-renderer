#include "TranslateArrowGizmo.hpp"

#include "core/passes/overlaygeometry/OverlayInstance.hpp"

#include "core/scene/Camera.hpp"
#include "core/math/LinearAlgebraHelpers.hpp"
#include "core/editor/command/TranslatePointsCommand.hpp"

#include <array>
#include <imgui.h>
#include <memory>

namespace lr
{

namespace
{

constexpr float kArrowRadius     = 0.05f;
constexpr float kArrowLength     = 1.0f;
constexpr float kOccludedOpacity = 0.3f;

// Indexed by TranslateArrowGizmoAxis. Position is left at its default and is
// set per-frame by whoever owns the gizmo (via getInstance()), since it
// tracks the selection centroid rather than being fixed at construction.
constexpr std::array<OverlayInstance, 4> kAxisInstances = {{OverlayInstance{
                                                                // +X (red): euler Z=-90 rotates +Y to +X
                                                                .primitive    = OverlayPrimitive::Arrow,
                                                                .position     = {0.0f, 0.0f, 0.0f},
                                                                .eulerDegrees = {0.0f, 0.0f, -90.0f},
                                                                .scale = {kArrowRadius, kArrowLength, kArrowRadius},
                                                                .color = {1.0f, 0.0f, 0.0f},
                                                                .occludedOpacity = kOccludedOpacity,
                                                            },
                                                            OverlayInstance{
                                                                // +Y (green): no rotation needed
                                                                .primitive    = OverlayPrimitive::Arrow,
                                                                .position     = {0.0f, 0.0f, 0.0f},
                                                                .eulerDegrees = {0.0f, 0.0f, 0.0f},
                                                                .scale = {kArrowRadius, kArrowLength, kArrowRadius},
                                                                .color = {0.0f, 1.0f, 0.0f},
                                                                .occludedOpacity = kOccludedOpacity,
                                                            },
                                                            OverlayInstance{
                                                                // +Z (blue): euler X=+90 rotates +Y to +Z
                                                                .primitive    = OverlayPrimitive::Arrow,
                                                                .position     = {0.0f, 0.0f, 0.0f},
                                                                .eulerDegrees = {90.0f, 0.0f, 0.0f},
                                                                .scale = {kArrowRadius, kArrowLength, kArrowRadius},
                                                                .color = {0.0f, 0.0f, 1.0f},
                                                                .occludedOpacity = kOccludedOpacity,
                                                            }}};

constexpr std::array<glm::vec3, 3> kAxisVectors = {{
    glm::vec3{1.0f, 0.0f, 0.0f},
    glm::vec3{0.0f, 1.0f, 0.0f},
    glm::vec3{0.0f, 0.0f, 1.0f},
}};

} // namespace

TranslateArrowGizmo::TranslateArrowGizmo(TranslateArrowGizmoAxis axis, const SceneObject &camera,
                                         const InputHandler &input, VertexManager &vertexManager,
                                         SelectionManager &selectionManager, CommandManager &commandManager)
    : Gizmo(kAxisInstances[static_cast<size_t>(axis)]), m_camera(camera), m_input(input),
      m_axis(kAxisVectors[static_cast<size_t>(axis)]), m_vertexManager(vertexManager),
      m_selectionManager(selectionManager), m_commandManager(commandManager)
{}

void TranslateArrowGizmo::onMouseDown(double ndcX, double ndcY, double aspect)
{
    // In here, we want to set the m_draggingOrigin to the closest point on the
    // axis line (the line originating at the gizmo centroid, going in the direction of the axis)
    // to the mouse ray in world space. This is what will be used as reference for translation
    const glm::mat4 vp                 = m_camera.getComponent<Camera>().viewProjectionMatrix(aspect);
    const glm::mat4 invVP              = glm::inverse(vp);
    const glm::vec4 pNear              = invVP * glm::vec4(ndcX, ndcY, 0.0f, 1.0f);
    const glm::vec4 pFar               = invVP * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
    const glm::vec3 cameraRayOrigin    = glm::vec3(pNear) / pNear.w;
    const glm::vec3 cameraRayDirection = glm::normalize(glm::vec3(pFar) / pFar.w - cameraRayOrigin);

    m_currentDraggingOrigin =
        math::closestPointOnLineToLine(m_instance.position, m_axis, cameraRayOrigin, cameraRayDirection);
    m_draggingOrigin = m_currentDraggingOrigin;
}

void TranslateArrowGizmo::onMouseUp(double ndcX, double ndcY, double aspect)
{
    m_commandManager.appendCommandWithoutExecuting(std::make_unique<TranslatePointsCommand>(
        m_vertexManager, m_selectionManager.getSelectedIndices(), m_currentDraggingOrigin - m_draggingOrigin));
}

void TranslateArrowGizmo::dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect)
{
    const glm::mat4 vp                 = m_camera.getComponent<Camera>().viewProjectionMatrix(aspect);
    const glm::mat4 invVP              = glm::inverse(vp);
    const glm::vec4 pNear              = invVP * glm::vec4(ndcX, ndcY, 0.0f, 1.0f);
    const glm::vec4 pFar               = invVP * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
    const glm::vec3 cameraRayOrigin    = glm::vec3(pNear) / pNear.w;
    const glm::vec3 cameraRayDirection = glm::normalize(glm::vec3(pFar) / pFar.w - cameraRayOrigin);

    glm::vec3 closestPointOnAxis =
        math::closestPointOnLineToLine(m_currentDraggingOrigin, m_axis, cameraRayOrigin, cameraRayDirection);
    glm::vec3 translation = closestPointOnAxis - m_currentDraggingOrigin;

    m_vertexManager.translateSelectedVertices(m_selectionManager.getSelectedIndices(), translation);
    m_currentDraggingOrigin = closestPointOnAxis; // Update the dragging origin for the next frame
}

} // namespace lr
