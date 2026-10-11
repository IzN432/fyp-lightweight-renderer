#pragma once

#include "core/scene/SceneObjectId.hpp"
#include "features/linear_blend_skinning/Joint.hpp"

#include <variant>

namespace lr
{

// Editor-only reference to one visible bone shaft. A bone is located by its child joint because every
// joint has at most one parent while one parent may have several outgoing bones. Roots have no
// incoming bone and therefore cannot be named by BoneRef.
struct BoneRef
{
    SceneObjectId skinObject{};
    JointIndex    childJoint = 0;

    friend bool operator==(const BoneRef &, const BoneRef &) = default;
};

// Editor-only reference to one joint ball. Unlike a bone, every palette entry has a joint ball,
// including roots and joints at branching points.
struct JointRef
{
    SceneObjectId skinObject{};
    JointIndex    joint = 0;

    friend bool operator==(const JointRef &, const JointRef &) = default;
};

// The two things pose mode can select. Both ultimately manipulate an existing joint SceneObject:
// a bone shaft targets its child joint, while a joint ball targets that joint directly.
using SkeletonSelection = std::variant<BoneRef, JointRef>;

} // namespace lr
