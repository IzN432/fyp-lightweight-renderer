#include "features/linear_blend_skinning/editor/SkeletonTopology.hpp"

#include "core/scene/Scene.hpp"
#include "features/linear_blend_skinning/Skin.hpp"

#include <stdexcept>
#include <unordered_map>

namespace lr
{

SkeletonTopology::SkeletonTopology(SceneObjectId skinObject, const Scene &scene, const Skin &skin)
    : m_skinObject(skinObject)
{
    std::unordered_map<SceneObjectId, JointIndex> jointByObject;
    jointByObject.reserve(skin.joints().size());

    for (JointIndex index = 0; index < skin.joints().size(); ++index)
    {
        const SceneObjectId object = skin.joints()[index].sceneObject;
        if (!scene.contains(object))
        {
            throw std::invalid_argument("Skeleton topology contains a missing joint object");
        }
        if (!jointByObject.emplace(object, index).second)
        {
            throw std::invalid_argument("Skeleton topology contains a duplicate joint object");
        }
    }

    for (JointIndex index = 0; index < skin.joints().size(); ++index)
    {
        const SceneObjectId childObject = skin.joints()[index].sceneObject;
        const std::optional<SceneObjectId> sceneParent = scene.getSceneObject(childObject).parent();

        if (sceneParent)
        {
            const auto parent = jointByObject.find(*sceneParent);
            if (parent != jointByObject.end())
            {
                m_bonesByChildJoint.emplace(index, SkeletonBone{
                    .parentJoint  = parent->second,
                    .parentObject = *sceneParent,
                    .childJoint   = index,
                    .childObject  = childObject,
                });
            }
        }
    }
}

const SkeletonBone *SkeletonTopology::find(BoneRef bone) const
{
    if (bone.skinObject != m_skinObject)
    {
        return nullptr;
    }
    const auto found = m_bonesByChildJoint.find(bone.childJoint);
    return found == m_bonesByChildJoint.end() ? nullptr : &found->second;
}

} // namespace lr
