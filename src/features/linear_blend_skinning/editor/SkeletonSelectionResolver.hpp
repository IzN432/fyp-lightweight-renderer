#pragma once

#include "core/scene/SceneObjectId.hpp"
#include "features/linear_blend_skinning/editor/SkeletonSelection.hpp"

#include <optional>

namespace lr
{

class Scene;
class Skin;

// Resolves a selected joint ball to its scene object. Bone shafts are not transforms and return
// null: the editor must first resolve a picked shaft to either endpoint. Also returns null when the
// selection belongs to another skin or has become stale.
std::optional<SceneObjectId> resolveSkeletonSelection(const Scene &scene, const Skin &skin,
                                                      const SkeletonSelection &selection);

} // namespace lr
