#pragma once

#include "core/scene/SceneObjectId.hpp"
#include "features/linear_blend_skinning/editor/SkeletonSelection.hpp"

#include <cstdint>
#include <optional>

namespace lr
{

class Scene;
class Skin;

// Bone shafts are visual hierarchy edges rather than scene objects. Keeping their posing rule here
// makes it independent of picking and gizmo implementations: both translation and rotation edit the
// child joint's local transform. Rotation therefore rotates the child joint and its descendants; it
// does not rotate the parent or orbit the child around the parent.
enum class BoneShaftPoseSemantics
{
    ChildJointLocalTransform,
};

struct SkeletonManipulationTarget
{
    SceneObjectId object;
    uint32_t      joint;
    bool          selectedThroughBone = false;
};

class SkeletonManipulationAdapter
{
public:
    static constexpr BoneShaftPoseSemantics boneSemantics =
        BoneShaftPoseSemantics::ChildJointLocalTransform;

    // A joint ball maps to its joint. A bone shaft maps to the joint named by BoneRef::childJoint.
    // Stale selections, selections from another shared skin, and missing scene objects do not map.
    static std::optional<SkeletonManipulationTarget> resolve(
        const Scene &scene, const Skin &skin, const SkeletonSelection &selection);
};

} // namespace lr
