#pragma once

#include <imgui.h>

#include <span>

namespace lr::gui
{

struct AnimationTrackResult
{
    bool progressChanged = false;
    bool viewChanged     = false;
};

// Draws an interactive animation timeline. Progress and keyframe values are normalized to [0, 1].
// viewHalfWidth is the distance from viewCenter to either visible edge, also in progress units.
// snapThreshold is a fraction of the currently visible viewport width.
AnimationTrackResult animationTrack(const char *label, float *progress, std::span<const float> keyframes,
                                    float *viewCenter, float *viewHalfWidth, float snapThreshold = 0.01f,
                                    ImVec2 size = ImVec2(-1.0f, 36.0f));

} // namespace lr::gui
