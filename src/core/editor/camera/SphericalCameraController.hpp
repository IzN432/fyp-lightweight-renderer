#pragma once

#include "CameraController.hpp"

#include <glm/glm.hpp>

namespace lr
{
    
class SphericalCameraController : public CameraController
{
public:
    SphericalCameraController(SceneObject &camera, InputHandler &input)
        : CameraController(camera, input) {}
    virtual ~SphericalCameraController() = default;

    void update(float dt) override;
private:
    // CAMERA — spherical orbit state (Blender-style)
    glm::vec3 m_orbitTarget{0.0f};
    float m_orbitRadius    = 5.0f;
    float m_orbitAzimuth   = 0.0f;   // radians; 0 = camera on +Z axis
    float m_orbitElevation = 0.0f;   // radians; 0 = horizontal
};

}