#include "features/animation/AnimatorComponent.hpp"

#include "core/app/ImGuiWidgets.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace lr
{

AnimatorComponent::AnimatorComponent(std::vector<AnimationClip> clips)
    : Component("AnimatorComponent"), m_clips(std::move(clips))
{}

void AnimatorComponent::play(size_t clipIndex)
{
    if (clipIndex >= m_clips.size())
    {
        throw std::out_of_range("AnimatorComponent clip index is out of range");
    }
    m_activeClip = clipIndex;
    m_selectedTrack = 0;
    m_playbackSeconds = 0.0f;
    m_playing = true;
    markDirty();
}

void AnimatorComponent::stop()
{
    m_playing = false;
    m_activeClip.reset();
    m_playbackSeconds = 0.0f;
    markDirty();
}

void AnimatorComponent::seek(float seconds)
{
    if (!m_activeClip)
    {
        return;
    }
    if (!std::isfinite(seconds))
    {
        throw std::invalid_argument("AnimatorComponent playback time must be finite");
    }

    const AnimationClip &clip = m_clips[m_activeClip.value()];
    m_playbackSeconds = std::clamp(seconds, 0.0f, clip.durationSeconds());
    for (const AnimationChannel &channel : clip.tracks())
    {
        std::visit([&](const auto &track) {
            track.apply(getOwningObject().scene(), m_playbackSeconds);
        }, channel);
    }
    markDirty();
}

void AnimatorComponent::setSpeedMultiplier(float speedMultiplier)
{
    if (!std::isfinite(speedMultiplier) || speedMultiplier < 0.0f)
    {
        throw std::invalid_argument("AnimatorComponent speed multiplier must be finite and non-negative");
    }
    m_speedMultiplier = speedMultiplier;
    markDirty();
}

void AnimatorComponent::update(float deltaSeconds)
{
    if (!m_playing || !m_activeClip)
    {
        return;
    }
    if (!std::isfinite(deltaSeconds) || deltaSeconds < 0.0f)
    {
        throw std::invalid_argument("AnimatorComponent delta time must be finite and non-negative");
    }

    const AnimationClip &clip = m_clips[m_activeClip.value()];
    const float duration = clip.durationSeconds();
    m_playbackSeconds += deltaSeconds * m_speedMultiplier;
    if (duration <= 0.0f)
    {
        m_playing = false;
    }
    else if (m_playbackSeconds >= duration)
    {
        if (m_loop)
        {
            m_playbackSeconds = std::fmod(m_playbackSeconds, duration);
        }
        else
        {
            m_playbackSeconds = duration;
            m_playing = false;
        }
    }

    seek(m_playbackSeconds);
}

void AnimatorComponent::onGUIImpl()
{
    if (m_clips.empty())
    {
        ImGui::TextDisabled("No animation clips");
        return;
    }

    for (size_t index = 0; index < m_clips.size(); ++index)
    {
        ImGui::PushID(static_cast<int>(index));
        const bool active = m_activeClip && m_activeClip.value() == index;
        if (ImGui::Selectable(m_clips[index].name().c_str(), active))
        {
            play(index);
        }
        ImGui::PopID();
    }

    if (m_activeClip)
    {
        const AnimationClip &clip = m_clips[m_activeClip.value()];
        ImGui::Text("Time: %.3f / %.3f s", m_playbackSeconds, clip.durationSeconds());
        if (ImGui::Checkbox("Loop", &m_loop))
        {
            markDirty();
        }
        float speedMultiplier = m_speedMultiplier;
        if (ImGui::DragFloat("Speed", &speedMultiplier, 0.05f, 0.0f, 10.0f, "%.2fx"))
        {
            setSpeedMultiplier(speedMultiplier);
        }
        if (m_playing)
        {
            if (ImGui::Button("Pause"))
            {
                pause();
            }
        }
        else if (ImGui::Button("Resume"))
        {
            m_playing = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Stop"))
        {
            stop();
        }

        if (ImGui::TreeNodeEx("Tracks", ImGuiTreeNodeFlags_DefaultOpen))
        {
            const float duration = clip.durationSeconds();
            float progress = duration > 0.0f ? m_playbackSeconds / duration : 0.0f;
            Scene &scene = getOwningObject().scene();

            const auto trackLabel = [&](const AnimationChannel &channel) {
                return std::visit([&](const auto &track) {
                    std::string objectName = "Missing object";
                    if (scene.contains(track.target()))
                    {
                        const SceneObject &target = scene.getSceneObject(track.target());
                        objectName = target.name.empty() ? "Scene Object " + std::to_string(track.target())
                                                         : target.name;
                    }

                    const char *propertyName = "Scale";
                    using Track = std::decay_t<decltype(track)>;
                    if constexpr (std::is_same_v<Track, TranslationTrack>)
                    {
                        propertyName = "Translation";
                    }
                    else if constexpr (std::is_same_v<Track, RotationTrack>)
                    {
                        propertyName = "Rotation";
                    }
                    return objectName + " (" + propertyName + ")";
                }, channel);
            };

            m_selectedTrack = std::min(m_selectedTrack, clip.tracks().size() - 1);
            const std::string selectedLabel = trackLabel(clip.tracks()[m_selectedTrack]);
            if (ImGui::BeginCombo("Track", selectedLabel.c_str()))
            {
                for (size_t trackIndex = 0; trackIndex < clip.tracks().size(); ++trackIndex)
                {
                    const std::string label = trackLabel(clip.tracks()[trackIndex]);
                    const bool selected = trackIndex == m_selectedTrack;
                    ImGui::PushID(static_cast<int>(trackIndex));
                    if (ImGui::Selectable(label.c_str(), selected))
                    {
                        m_selectedTrack = trackIndex;
                    }
                    if (selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                    ImGui::PopID();
                }
                ImGui::EndCombo();
            }

            const AnimationChannel &selectedChannel = clip.tracks()[m_selectedTrack];
            std::visit([&](const auto &track) {
                    std::vector<float> keyframes;
                    keyframes.reserve(track.keyframes().size());
                    for (const auto &keyframe : track.keyframes())
                    {
                        keyframes.push_back(duration > 0.0f ? keyframe.seconds / duration : 0.0f);
                    }

                    const gui::AnimationTrackResult trackResult = gui::animationTrack(
                        "##track", &progress, keyframes, &m_trackViewCenter, &m_trackViewHalfWidth, 0.01f,
                        ImVec2(ImGui::GetContentRegionAvail().x, 36.0f));
                    if (trackResult.progressChanged)
                    {
                        pause();
                        seek(progress * duration);
                    }
                }, selectedChannel);
            ImGui::TreePop();
        }
    }
}

} // namespace lr
