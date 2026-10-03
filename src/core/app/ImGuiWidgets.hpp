#pragma once

#include <imgui.h>

#include <span>

namespace lr::gui
{

// Draws an interactive animation timeline. Progress and keyframe values are normalized to [0, 1].
// Returns true when user interaction changes progress.
bool animationTrack(const char *label, float *progress, std::span<const float> keyframes,
                    float snapThreshold = 0.01f, ImVec2 size = ImVec2(-1.0f, 36.0f));

} // namespace lr::gui
