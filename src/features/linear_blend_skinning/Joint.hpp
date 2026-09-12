#pragma once

#include "features/linear_blend_skinning/SkeletonNode.hpp"

#include <glm/glm.hpp>

#include <cstdint>

namespace lr
{

using JointIndex = uint32_t;

// One entry in a skin's joint palette. Vertex influence indices address joints
// in palette order. The animated transform belongs to the referenced node.
struct Joint
{
    SkeletonNodeIndex node = 0;
    glm::mat4         inverseBindMatrix{1.0f};
};

} // namespace lr
