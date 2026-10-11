#pragma once

#include "features/linear_blend_skinning/editor/SkeletonSelection.hpp"

#include <map>

namespace lr
{

class Scene;
class Skin;

// Read-only editor view derived from a Skin palette and the Scene hierarchy. It owns IDs and
// indices, never SceneObject pointers, so rebuilding it at scene-change boundaries is safe.
struct SkeletonBone
{
    JointIndex    parentJoint = 0;
    SceneObjectId parentObject{};
    JointIndex    childJoint = 0;
    SceneObjectId childObject{};
};

class SkeletonTopology
{
public:
    SkeletonTopology(const Scene &scene, const Skin &skin);

    SkinId skinId() const { return m_skinId; }
    const std::map<JointIndex, SkeletonBone> &bonesByChildJoint() const { return m_bonesByChildJoint; }

    // A bone exists only for a joint whose scene parent is also in this skin. Joint balls resolve
    // directly through Skin::joints(); this topology does not duplicate the joint palette.
    const SkeletonBone *find(BoneRef bone) const;

private:
    SkinId                             m_skinId{};
    std::map<JointIndex, SkeletonBone> m_bonesByChildJoint;
};

} // namespace lr
