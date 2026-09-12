#pragma once

#include "core/scene/Transform.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace lr
{

using SkeletonNodeIndex = uint32_t;

// A transform node in a skeleton hierarchy. Non-joint ancestors must also be
// retained because their transforms affect descendant joints.
struct SkeletonNode
{
    std::string                      name;
    uint32_t                         sourceNodeIndex = 0;
    std::optional<SkeletonNodeIndex> parent;
    Transform                        localTransform;
};

} // namespace lr
