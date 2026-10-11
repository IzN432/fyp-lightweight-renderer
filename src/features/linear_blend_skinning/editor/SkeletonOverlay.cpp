#include "features/linear_blend_skinning/editor/SkeletonOverlay.hpp"

#include "core/scene/Scene.hpp"
#include "features/linear_blend_skinning/SkinComponent.hpp"
#include "features/linear_blend_skinning/editor/SkeletonTopology.hpp"

#include <glm/common.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cmath>
#include <stdexcept>

namespace lr
{

namespace
{
glm::vec3 worldPosition(const Scene &scene, SceneObjectId object)
{
    return glm::vec3(scene.getSceneObject(object).worldMatrix()[3]);
}

glm::quat rotatePositiveYTo(const glm::vec3 &direction)
{
    constexpr glm::vec3 source(0.0f, 1.0f, 0.0f);
    const float cosine = glm::dot(source, direction);
    if (cosine < -0.999999f)
    {
        return glm::angleAxis(glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
    }
    const glm::vec3 axis = glm::cross(source, direction);
    return glm::normalize(glm::quat(1.0f + cosine, axis.x, axis.y, axis.z));
}
} // namespace

std::vector<OverlayInstance> buildSkeletonOverlay(const Scene &scene, SceneObjectId skinObject,
                                                  const SkeletonOverlayStyle &style)
{
    if (!scene.contains(skinObject))
    {
        throw std::invalid_argument("Cannot visualize a skin object that is not in the scene");
    }

    const SceneObject &object = scene.getSceneObject(skinObject);
    if (!object.hasComponent<SkinComponent>())
    {
        throw std::invalid_argument("Cannot visualize an object without a skin component");
    }
    if (style.boneRadiusRatio <= 0.0f || style.jointRadiusRatio <= 0.0f)
    {
        throw std::invalid_argument("Skeleton overlay size ratios must be positive");
    }

    const Skin             &skin = object.getComponent<SkinComponent>().skin();
    const SkeletonTopology  topology(skinObject, scene, skin);

    float  totalLength = 0.0f;
    size_t lengthCount = 0;
    for (const auto &[childJoint, bone] : topology.bonesByChildJoint())
    {
        const float length = glm::length(worldPosition(scene, bone.childObject) -
                                         worldPosition(scene, bone.parentObject));
        if (length > 1e-6f)
        {
            totalLength += length;
            ++lengthCount;
        }
    }
    const float referenceLength = lengthCount == 0 ? 1.0f : totalLength / static_cast<float>(lengthCount);
    const float boneRadius      = referenceLength * style.boneRadiusRatio;
    const float jointRadius     = referenceLength * style.jointRadiusRatio;

    std::vector<OverlayInstance> instances;
    instances.reserve(topology.bonesByChildJoint().size() + skin.joints().size());

    for (const auto &[childJoint, bone] : topology.bonesByChildJoint())
    {
        const glm::vec3 start = worldPosition(scene, bone.parentObject);
        const glm::vec3 delta = worldPosition(scene, bone.childObject) - start;
        const float     length = glm::length(delta);
        if (length <= 1e-6f)
        {
            continue;
        }

        // The canonical arrow runs from y=0 to y=1 and has a 0.05-unit shaft radius.
        const glm::quat orientation = rotatePositiveYTo(delta / length);
        instances.push_back({
            .primitive       = OverlayPrimitive::Arrow,
            .position        = start,
            .eulerDegrees    = glm::degrees(glm::eulerAngles(orientation)),
            .scale           = {boneRadius / 0.05f, length, boneRadius / 0.05f},
            .color           = style.boneColor,
            .occludedOpacity = style.occludedOpacity,
        });
    }

    for (const Joint &joint : skin.joints())
    {
        instances.push_back({
            .primitive       = OverlayPrimitive::Sphere,
            .position        = worldPosition(scene, joint.sceneObject),
            .scale           = glm::vec3(jointRadius),
            .color           = style.jointColor,
            .occludedOpacity = style.occludedOpacity,
        });
    }
    return instances;
}

} // namespace lr
