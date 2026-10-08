#pragma once

#include "core/scene/Component.hpp"

#include <string>
#include <utility>

namespace lr
{

// Editor-facing contract for the controller component attached to the main camera. Concrete
// controllers own their navigation model; the editor only needs a common way to restore its
// default orbit through the shortcut system.
class CameraController : public Component
{
public:
    explicit CameraController(std::string name) : Component(std::move(name)) {}

    virtual void resetTransformation() = 0;
};

} // namespace lr
