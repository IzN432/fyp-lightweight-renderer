#pragma once

#include "core/scene/Component.hpp"

#include <glm/glm.hpp>

namespace lr
{

class InputHandler;

class SphericalCameraController : public Component
{
public:
    SphericalCameraController() : Component("Spherical Camera Controller") {}

    // Decides for itself whether the UI has the pointer. For a host that arbitrates pointer
    // priority — see EditorInputRouter::viewportNavigationAllowed() — use the overload below.
    void update(InputHandler &input, float dt);

    // `navigationAllowed` is the host's verdict on whether orbit/pan/zoom may act this frame. The
    // controller does not ask who or what is holding the pointer; that is not its decision to make.
    void update(InputHandler &input, float dt, bool navigationAllowed);

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

    void onGUIImpl() override;

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
