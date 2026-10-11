#include "features/linear_blend_skinning/editor/SkeletonSelectionResolver.hpp"

#include "core/scene/Scene.hpp"
#include "features/linear_blend_skinning/Skin.hpp"
#include <type_traits>

namespace lr
{

std::optional<SceneObjectId> resolveSkeletonSelection(const Scene &scene, const Skin &skin,
                                                      const SkeletonSelection &selection)
{
    return std::visit([&](const auto &selected) -> std::optional<SceneObjectId> {
        using Selection = std::decay_t<decltype(selected)>;
        if (selected.skin != skin.id())
        {
            return std::nullopt;
        }

        SceneObjectId object{};
        if constexpr (std::is_same_v<Selection, BoneRef>)
        {
            return std::nullopt;
        }
        else
        {
            if (selected.joint >= skin.joints().size())
            {
                return std::nullopt;
            }
            object = skin.joints()[selected.joint].sceneObject;
        }
        return scene.contains(object) ? std::optional<SceneObjectId>(object) : std::nullopt;
    }, selection);
}

} // namespace lr
