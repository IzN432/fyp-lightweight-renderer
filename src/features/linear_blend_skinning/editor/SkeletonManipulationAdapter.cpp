#include "features/linear_blend_skinning/editor/SkeletonManipulationAdapter.hpp"

#include "core/scene/Scene.hpp"
#include "features/linear_blend_skinning/Skin.hpp"

#include <type_traits>

namespace lr
{

std::optional<SkeletonManipulationTarget> SkeletonManipulationAdapter::resolve(
    const Scene &scene, const Skin &skin, const SkeletonSelection &selection)
{
    return std::visit(
        [&](const auto &selected) -> std::optional<SkeletonManipulationTarget> {
            using Selection = std::decay_t<decltype(selected)>;
            if (selected.skin != skin.id())
            {
                return std::nullopt;
            }

            const uint32_t joint = [&] {
                if constexpr (std::is_same_v<Selection, BoneRef>)
                    return selected.childJoint;
                else
                    return selected.joint;
            }();
            if (joint >= skin.joints().size())
            {
                return std::nullopt;
            }

            const SceneObjectId object = skin.joints()[joint].sceneObject;
            if (!scene.contains(object))
            {
                return std::nullopt;
            }
            return SkeletonManipulationTarget{
                .object              = object,
                .joint               = joint,
                .selectedThroughBone = std::is_same_v<Selection, BoneRef>,
            };
        },
        selection);
}

} // namespace lr
