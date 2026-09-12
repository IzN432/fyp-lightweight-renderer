#pragma once

#include "features/linear_blend_skinning/Joint.hpp"

#include <optional>
#include <string>
#include <vector>

namespace lr
{

struct Skin
{
    std::string                      name;
    std::vector<SkeletonNode>        nodes;
    std::vector<Joint>               joints;
};

} // namespace lr
