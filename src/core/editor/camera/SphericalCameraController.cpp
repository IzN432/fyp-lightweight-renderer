#include "SphericalCameraController.hpp"

#include "core/app/InputHandler.hpp"
#include "core/editor/PointerCapture.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"

#include <glm/glm.hpp>

#include <algorithm>

namespace lr
{

void SphericalCameraController::update(InputHandler &input, float dt)
{
    update(input, dt, !imguiCapturesPointer());
}

void SphericalCameraController::update(InputHandler &input, float, bool navigationAllowed)
{
    double dx, dy;
    input.getMouseDelta(dx, dy);
    double scroll = input.getScrollDelta();
    bool   mmb    = input.isMouseButtonPressed(GLFW_MOUSE_BUTTON_MIDDLE);
    bool   shift  = input.isKeyPressed(GLFW_KEY_LEFT_SHIFT) || input.isKeyPressed(GLFW_KEY_RIGHT_SHIFT);

    bool reset = input.isKeyPressed(GLFW_KEY_R);
    if (reset)
    {
        m_orbitTarget    = glm::vec3(0.0f);
        m_orbitRadius    = 5.0f;
        m_orbitAzimuth   = 0.0f;
        m_orbitElevation = 0.0f;
    }

    if (navigationAllowed)
    {
        if (mmb && shift)
        {
            // Pan: translate the target in the camera's right/up plane.
            auto &t        = getOwningObject().getComponent<TransformComponent>();
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
            m_orbitRadius = std::max(m_orbitRadius, 0.01f);
        }
    }

    applyPose();
}

void SphericalCameraController::onGUIImpl()
{
    bool changed = false;
    changed |= ImGui::DragFloat3("Target", &m_orbitTarget.x, 0.01f);
    changed |= ImGui::DragFloat("Radius", &m_orbitRadius, 0.01f, 0.01f, 1000.0f);
    changed |= ImGui::DragFloat("Azimuth (radians)", &m_orbitAzimuth, 0.01f);
    changed |= ImGui::SliderFloat("Elevation (radians)", &m_orbitElevation,
                                  glm::radians(-89.0f), glm::radians(89.0f));
    if (changed)
    {
        m_orbitRadius = std::max(m_orbitRadius, 0.01f);
        m_orbitElevation = glm::clamp(m_orbitElevation, glm::radians(-89.0f), glm::radians(89.0f));
        applyPose();
        markDirty();
    }
}

void SphericalCameraController::setOrbitState(const OrbitState &state)
{
    restoreOrbitState(state);
    applyPose();
}

void SphericalCameraController::restoreOrbitState(const OrbitState &state)
{
    m_orbitTarget    = state.target;
    m_orbitRadius    = std::max(state.radius, 0.01f);
    m_orbitAzimuth   = state.azimuth;
    m_orbitElevation = glm::clamp(state.elevation, glm::radians(-89.0f), glm::radians(89.0f));
}

void SphericalCameraController::onLoaded() { applyPose(); }

void SphericalCameraController::applyPose()
{
    glm::vec3 pos(m_orbitTarget.x + m_orbitRadius * std::cos(m_orbitElevation) * std::sin(m_orbitAzimuth),
                  m_orbitTarget.y + m_orbitRadius * std::sin(m_orbitElevation),
                  m_orbitTarget.z + m_orbitRadius * std::cos(m_orbitElevation) * std::cos(m_orbitAzimuth));

    getOwningObject().getComponent<TransformComponent>().setPosition(pos);
    const glm::mat4 view = glm::lookAt(pos, m_orbitTarget, glm::vec3(0.0f, 1.0f, 0.0f));
    getOwningObject().getComponent<TransformComponent>().setRotation(glm::conjugate(glm::quat_cast(view)));
}

} // namespace lr
