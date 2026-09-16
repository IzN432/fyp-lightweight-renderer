#pragma once

#include "core/passes/overlaylines/OverlayLine.hpp"

#include <vector>

namespace lr
{

class Scene;

// Produces non-pickable overlay instances for every visible ColliderComponent in the scene.
std::vector<OverlayLine> buildColliderOverlayLines(const Scene &scene);

} // namespace lr
