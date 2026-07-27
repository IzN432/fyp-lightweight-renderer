#pragma once

#include "core/scene/Camera.hpp"
#include "core/app/InputHandler.hpp"

#include <memory>

namespace lr
{

class CameraController
{
    // this class should take in a reference to the camera object in the scene, then it has virtual callback function
    // for update that takes in a reference to the input system and delta time and does the camera transformations
    // accordingly.
public:
    CameraController(SceneObject &camera, InputHandler &input)
        : m_cameraSceneObject(camera), m_input(input) {}
    virtual ~CameraController() = default;

    virtual void update(float dt) = 0;

protected:
    // reference to the camera object in the scene
    SceneObject  &m_cameraSceneObject;
    InputHandler &m_input;
};

} // namespace lr
