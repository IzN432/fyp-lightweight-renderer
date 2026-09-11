#pragma once

#include <glm/vec3.hpp>

#include <span>
#include <vector>

namespace lr
{

// Normalizes a non-negative scalar field to [0, 1], clamping the upper tail at upperPercentile
// so isolated outliers do not flatten the visible range, then applies the renderer's four-band
// blue -> cyan -> yellow -> red ramp.
std::vector<glm::vec3> makeHeatmapColors(std::span<const float> values, float upperPercentile = 0.95f);

} // namespace lr
