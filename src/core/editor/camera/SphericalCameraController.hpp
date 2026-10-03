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

    // The orbit: the camera looks at `target` from `radius` away, `azimuth` radians around +Y (0 = on
    // the +Z axis) and `elevation` radians above the horizontal. Setting it places the camera at once,
    // e.g. to frame a scene; update() then continues from there.
    struct OrbitState
    {
        glm::vec3 target{0.0f};
        float     radius    = 5.0f;
        float     azimuth   = 0.0f;
        float     elevation = 0.0f;
    };
    OrbitState orbitState() const { return {m_orbitTarget, m_orbitRadius, m_orbitAzimuth, m_orbitElevation}; }
    void       setOrbitState(const OrbitState &state);

private:
    // Places the camera object according to the orbit state.
    void applyPose();

    // CAMERA — spherical orbit state (Blender-style)
    glm::vec3 m_orbitTarget{0.0f};
    float     m_orbitRadius    = 5.0f;
    float     m_orbitAzimuth   = 0.0f; // radians; 0 = camera on +Z axis
    float     m_orbitElevation = 0.0f; // radians; 0 = horizontal
};

} // namespace lr
