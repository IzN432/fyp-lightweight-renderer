#include "TranslateBoxGizmo.hpp"

#include "core/scene/Transform.hpp"
#include "core/scene/Camera.hpp"
#include "core/math/LinearAlgebraHelpers.hpp"

namespace lr
{

namespace
{

constexpr float kArrowRadius     = 0.05f;
constexpr float kOccludedOpacity = 0.3f;

OverlayInstance boxInstance = {
    // center cube (white): screen-plane drag
    .primitive = OverlayPrimitive::Cube, .position = {0.0f, 0.0f, 0.0f},
    .eulerDegrees = {0.0f, 0.0f, 0.0f},  .scale = {kArrowRadius * 0.45f, kArrowRadius * 0.45f, kArrowRadius * 0.45f},
    .color = {0.7f, 0.7f, 0.7f},         .occludedOpacity = kOccludedOpacity,
};

}

TranslateBoxGizmo::TranslateBoxGizmo(const SceneObject &camera, const InputHandler &input, VertexManager &vertexManager,
                                     SelectionManager &selectionManager)
    : Gizmo(boxInstance), m_camera(camera), m_input(input), m_vertexManager(vertexManager),
      m_selectionManager(selectionManager)
{}

void TranslateBoxGizmo::onMouseDown(double ndcX, double ndcY, double aspect) 
{
    // Define the m_draggingPlane as the plane perpendicular to the camera's forward direction and passing through the centroid of the selected vertices
    m_draggingPlane = math::planeFromNormalAndPoint(m_camera.getComponent<Transform>().forward(), m_instance.position);
    
    const glm::mat4 vp                 = m_camera.getComponent<Camera>().viewProjectionMatrix(aspect);
    const glm::mat4 invVP              = glm::inverse(vp);
    const glm::vec4 pNear              = invVP * glm::vec4(ndcX, ndcY, 0.0f, 1.0f);
    const glm::vec4 pFar               = invVP * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
    const glm::vec3 cameraRayOrigin    = glm::vec3(pNear) / pNear.w;
    const glm::vec3 cameraRayDirection = glm::normalize(glm::vec3(pFar) / pFar.w - cameraRayOrigin);
    
    m_draggingOrigin = math::intersectionBetweenRayAndPlane(cameraRayOrigin, cameraRayDirection, m_draggingPlane);
}

void TranslateBoxGizmo::onMouseUp(double ndcX, double ndcY, double aspect) 
{

}

void TranslateBoxGizmo::dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect) 
{
    const glm::mat4 vp                 = m_camera.getComponent<Camera>().viewProjectionMatrix(aspect);
    const glm::mat4 invVP              = glm::inverse(vp);
    const glm::vec4 pNear              = invVP * glm::vec4(ndcX, ndcY, 0.0f, 1.0f);
    const glm::vec4 pFar               = invVP * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
    const glm::vec3 cameraRayOrigin    = glm::vec3(pNear) / pNear.w;
    const glm::vec3 cameraRayDirection = glm::normalize(glm::vec3(pFar) / pFar.w - cameraRayOrigin);

    glm::vec3 intersection = math::intersectionBetweenRayAndPlane(cameraRayOrigin, cameraRayDirection, m_draggingPlane);
    glm::vec3 translation = intersection - m_draggingOrigin;
    m_vertexManager.translateSelectedVertices(m_selectionManager.getSelectedIndices(), translation);
    m_draggingOrigin = intersection;
}

} // namespace lr