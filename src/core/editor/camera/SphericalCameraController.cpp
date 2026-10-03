#include "SphericalCameraController.hpp"

#include "core/scene/TransformComponent.hpp"

#include <glm/glm.hpp>

namespace lr
{

void SphericalCameraController::update(float dt) { update(dt, false); }

void SphericalCameraController::update(float dt, bool gizmoCapturesPrimaryMouse)
{
    double dx, dy;
    m_input.getMouseDelta(dx, dy);
    double scroll = m_input.getScrollDelta();
    bool   mmb    = m_input.isMouseButtonPressed(GLFW_MOUSE_BUTTON_MIDDLE);
    bool   shift  = m_input.isKeyPressed(GLFW_KEY_LEFT_SHIFT) || m_input.isKeyPressed(GLFW_KEY_RIGHT_SHIFT);

    bool reset = m_input.isKeyPressed(GLFW_KEY_R);
    if (reset)
    {
        m_orbitTarget    = glm::vec3(0.0f);
        m_orbitRadius    = 5.0f;
        m_orbitAzimuth   = 0.0f;
        m_orbitElevation = 0.0f;
    }

    if (!ImGui::GetIO().WantCaptureMouse || gizmoCapturesPrimaryMouse)
    {
        if (mmb && shift)
        {
            // Pan: translate target in m_camera right/up plane
            auto &t        = m_cameraSceneObject.getComponent<TransformComponent>();
            float panSpeed = m_orbitRadius * 0.002f;
            m_orbitTarget -= t.transform().right() * (float)dx * panSpeed;
            m_orbitTarget += t.transform().up() * (float)dy * panSpeed;
        } else if (mmb)
        {
            m_orbitAzimuth -= (float)dx * 0.01f;
            m_orbitElevation -= (float)dy * 0.01f;
            m_orbitElevation = glm::clamp(m_orbitElevation, glm::radians(-89.0f), glm::radians(89.0f));
        }

        if (scroll != 0.0)
        {
            m_orbitRadius *= std::pow(1.0f / 1.1f, (float)scroll);
            m_orbitRadius = glm::clamp(m_orbitRadius, 0.01f, 1000.0f);
        }
    }

    applyPose();
}

void SphericalCameraController::setOrbitState(const OrbitState &state)
{
    m_orbitTarget    = state.target;
    m_orbitRadius    = glm::clamp(state.radius, 0.01f, 1000.0f);
    m_orbitAzimuth   = state.azimuth;
    m_orbitElevation = glm::clamp(state.elevation, glm::radians(-89.0f), glm::radians(89.0f));
    applyPose();
}

void SphericalCameraController::applyPose()
{
    glm::vec3 pos(m_orbitTarget.x + m_orbitRadius * std::cos(m_orbitElevation) * std::sin(m_orbitAzimuth),
                  m_orbitTarget.y + m_orbitRadius * std::sin(m_orbitElevation),
                  m_orbitTarget.z + m_orbitRadius * std::cos(m_orbitElevation) * std::cos(m_orbitAzimuth));

    m_cameraSceneObject.getComponent<TransformComponent>().setPosition(pos);
    const glm::mat4 view = glm::lookAt(pos, m_orbitTarget, glm::vec3(0.0f, 1.0f, 0.0f));
    m_cameraSceneObject.getComponent<TransformComponent>().setRotation(glm::conjugate(glm::quat_cast(view)));
}

} // namespace lr
