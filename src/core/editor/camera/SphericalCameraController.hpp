#pragma once

#include "CameraController.hpp"

#include <glm/glm.hpp>

namespace lr
{

class SphericalCameraController : public CameraController
{
public:
    SphericalCameraController(SceneObject &camera, InputHandler &input) : CameraController(camera, input) {}
    virtual ~SphericalCameraController() = default;

    void update(float dt) override;
    // ImGuizmo uses ImGui's global mouse-capture flag for its primary-button handles. Camera
    // orbit/pan/zoom remain valid viewport interactions in that case, while ordinary ImGui
    // windows must still block them.
    void update(float dt, bool gizmoCapturesPrimaryMouse);

private:
    // CAMERA — spherical orbit state (Blender-style)
    glm::vec3 m_orbitTarget{0.0f};
    float     m_orbitRadius    = 5.0f;
    float     m_orbitAzimuth   = 0.0f; // radians; 0 = camera on +Z axis
    float     m_orbitElevation = 0.0f; // radians; 0 = horizontal
};

} // namespace lr
