#include "core/app/ImGuiWidgets.hpp"

#include <algorithm>
#include <cmath>

namespace lr::gui
{
namespace
{

float snapToNearestKeyframe(float progress, std::span<const float> keyframes, float threshold)
{
    float snapped      = progress;
    float bestDistance = std::max(threshold, 0.0f);

    for (const float keyframe : keyframes)
    {
        if (!std::isfinite(keyframe))
        {
            continue;
        }

        const float candidate = std::clamp(keyframe, 0.0f, 1.0f);
        const float distance  = std::abs(candidate - progress);
        if (distance <= bestDistance)
        {
            bestDistance = distance;
            snapped      = candidate;
        }
    }

    return snapped;
}

} // namespace

AnimationTrackResult animationTrack(const char *label, float *progress, std::span<const float> keyframes,
                                    float *viewCenter, float *viewHalfWidth, float snapThreshold, ImVec2 size)
{
    IM_ASSERT(label != nullptr);
    IM_ASSERT(progress != nullptr);
    IM_ASSERT(viewCenter != nullptr);
    IM_ASSERT(viewHalfWidth != nullptr);

    AnimationTrackResult result;

    constexpr float minimumHalfWidth   = 0.001f;
    const float sanitizedHalfWidth = std::clamp(std::isfinite(*viewHalfWidth) ? *viewHalfWidth : 0.5f,
                                                minimumHalfWidth, 0.5f);
    const float sanitizedCenter =
        std::clamp(std::isfinite(*viewCenter) ? *viewCenter : 0.5f, sanitizedHalfWidth, 1.0f - sanitizedHalfWidth);
    if (sanitizedHalfWidth != *viewHalfWidth || sanitizedCenter != *viewCenter)
    {
        *viewHalfWidth    = sanitizedHalfWidth;
        *viewCenter       = sanitizedCenter;
        result.viewChanged = true;
    }

    const float defaultHeight = 36.0f;
    if (size.x == 0.0f)
    {
        size.x = ImGui::GetContentRegionAvail().x;
    }
    if (size.y <= 0.0f)
    {
        size.y = defaultHeight;
    }

    ImGui::InvisibleButton(label, size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
    ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);

    const ImVec2 itemMin = ImGui::GetItemRectMin();
    const ImVec2 itemMax = ImGui::GetItemRectMax();
    const float width    = std::max(itemMax.x - itemMin.x, 1.0f);

    ImGuiIO &io = ImGui::GetIO();
    if (ImGui::IsItemHovered() && io.MouseWheel != 0.0f)
    {
        const float zoomedHalfWidth = std::clamp(*viewHalfWidth * std::pow(0.8f, io.MouseWheel),
                                                 minimumHalfWidth, 0.5f);
        if (zoomedHalfWidth != *viewHalfWidth)
        {
            *viewHalfWidth     = zoomedHalfWidth;
            *viewCenter        = std::clamp(*viewCenter, *viewHalfWidth, 1.0f - *viewHalfWidth);
            result.viewChanged = true;
        }
    }

    const float visibleSpan = 2.0f * *viewHalfWidth;
    if (ImGui::IsItemActive() && ImGui::IsMouseDown(ImGuiMouseButton_Middle) && io.MouseDelta.x != 0.0f)
    {
        const float pannedCenter = std::clamp(*viewCenter - io.MouseDelta.x / width * visibleSpan,
                                              *viewHalfWidth, 1.0f - *viewHalfWidth);
        if (pannedCenter != *viewCenter)
        {
            *viewCenter        = pannedCenter;
            result.viewChanged = true;
        }
    }

    const float visibleMin = *viewCenter - *viewHalfWidth;
    const float visibleMax = *viewCenter + *viewHalfWidth;

    if (ImGui::IsItemActive() && ImGui::IsMouseDown(ImGuiMouseButton_Left))
    {
        const float mouseFraction = std::clamp((io.MousePos.x - itemMin.x) / width, 0.0f, 1.0f);
        float candidate = visibleMin + mouseFraction * visibleSpan;
        candidate       = snapToNearestKeyframe(candidate, keyframes, snapThreshold * visibleSpan);
        if (candidate != *progress)
        {
            *progress              = candidate;
            result.progressChanged = true;
        }
    }

    const float displayedProgress = std::clamp(std::isfinite(*progress) ? *progress : 0.0f, 0.0f, 1.0f);

    ImDrawList *drawList = ImGui::GetWindowDrawList();
    const float markerHeight = std::min(12.0f, size.y * 0.4f);
    const ImVec2 trackMin(itemMin.x, itemMin.y + markerHeight);
    const ImVec2 trackMax(itemMax.x, itemMax.y);
    const float rounding = std::min(3.0f, (trackMax.y - trackMin.y) * 0.2f);

    const ImU32 backgroundColor = ImGui::GetColorU32(ImVec4(0.20f, 0.21f, 0.23f, 1.0f));
    const ImU32 progressColor   = ImGui::GetColorU32(ImVec4(0.12f, 0.30f, 0.43f, 0.65f));
    const ImU32 borderColor     = ImGui::GetColorU32(
        ImGui::IsItemHovered() ? ImVec4(0.48f, 0.50f, 0.54f, 1.0f) : ImVec4(0.31f, 0.32f, 0.35f, 1.0f));
    const ImU32 keyframeColor = ImGui::GetColorU32(ImVec4(1.0f, 0.72f, 0.12f, 1.0f));
    const ImU32 playheadColor = ImGui::GetColorU32(ImVec4(0.20f, 0.52f, 0.95f, 1.0f));

    const float playheadX = itemMin.x + (displayedProgress - visibleMin) / visibleSpan * width;
    drawList->PushClipRect(itemMin, itemMax, true);
    drawList->AddRectFilled(trackMin, trackMax, backgroundColor, rounding);
    if (playheadX > trackMin.x)
    {
        drawList->AddRectFilled(trackMin, ImVec2(std::min(playheadX, trackMax.x), trackMax.y), progressColor,
                                rounding, ImDrawFlags_RoundCornersLeft);
    }
    drawList->AddRect(trackMin, trackMax, borderColor, rounding);

    const float trackCenterY = (trackMin.y + trackMax.y) * 0.5f;
    const float keyframeRadius = std::clamp((trackMax.y - trackMin.y) * 0.16f, 3.0f, 5.0f);
    for (const float keyframe : keyframes)
    {
        if (!std::isfinite(keyframe))
        {
            continue;
        }
        const float normalizedKeyframe = std::clamp(keyframe, 0.0f, 1.0f);
        if (normalizedKeyframe < visibleMin || normalizedKeyframe > visibleMax)
        {
            continue;
        }
        const float keyframeX = itemMin.x + (normalizedKeyframe - visibleMin) / visibleSpan * width;
        drawList->AddCircleFilled(ImVec2(keyframeX, trackCenterY), keyframeRadius, keyframeColor);
    }

    if (displayedProgress >= visibleMin && displayedProgress <= visibleMax)
    {
        drawList->AddLine(ImVec2(playheadX, itemMin.y + markerHeight - 1.0f), ImVec2(playheadX, itemMax.y),
                          playheadColor, 2.0f);

        const float halfMarkerWidth = 7.0f;
        const ImVec2 markerPoints[] = {
            ImVec2(playheadX - halfMarkerWidth, itemMin.y),
            ImVec2(playheadX + halfMarkerWidth, itemMin.y),
            ImVec2(playheadX + halfMarkerWidth, itemMin.y + markerHeight - 4.0f),
            ImVec2(playheadX, itemMin.y + markerHeight),
            ImVec2(playheadX - halfMarkerWidth, itemMin.y + markerHeight - 4.0f),
        };
        drawList->AddConvexPolyFilled(markerPoints, IM_ARRAYSIZE(markerPoints), playheadColor);
    }
    drawList->PopClipRect();

    return result;
}

} // namespace lr::gui
