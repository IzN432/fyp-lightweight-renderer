#pragma once

#include "core/scene/SceneObjectId.hpp"

#include <glm/glm.hpp>

#include <cstdint>

namespace lr
{

using JointIndex = uint32_t;

// One entry in a skin's joint palette. Vertex influence indices address joints
// in palette order. The transform belongs to the referenced scene object.
struct Joint
{
    SceneObjectId sceneObject{};
    glm::mat4     inverseBindMatrix{1.0f};
};

} // namespace lr
