#pragma once

#include "core/utility/Uuid.hpp"

namespace lr
{

// Scene objects are named by a generated identity rather than by their position in the scene, so a
// saved reference (an animation target, a skin joint, a parent link) still names the same object
// after a round trip, and two scenes can be loaded into one Scene without their ids colliding.
using SceneObjectId = Uuid;

} // namespace lr
