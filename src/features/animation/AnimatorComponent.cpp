#include "features/animation/AnimatorComponent.hpp"

#include "core/scene/SceneObject.hpp"

#include <imgui.h>

#include <cmath>
#include <stdexcept>
#include <utility>

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

    for (const AnimationChannel &channel : clip.tracks())
    {
        std::visit([&](const auto &track) {
            track.apply(getOwningObject().scene(), m_playbackSeconds);
        }, channel);
    }
    markDirty();
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
    }
}

} // namespace lr
