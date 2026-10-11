#include "features/animation/AnimationClip.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace lr
{

AnimationClip::AnimationClip(std::string name, std::vector<AnimationChannel> tracks,
                             std::optional<float> durationSeconds)
    : m_name(std::move(name)), m_tracks(std::move(tracks)), m_durationSeconds(durationSeconds)
{
    if (m_name.empty())
    {
        m_name = "Animation";
    }
    if (m_durationSeconds) setDurationSeconds(*m_durationSeconds);
}

void AnimationClip::setName(std::string name)
{
    m_name = name.empty() ? "Animation" : std::move(name);
}

float AnimationClip::durationSeconds() const
{
    if (m_durationSeconds)
    {
        return *m_durationSeconds;
    }
    float duration = 0.0f;
    for (const AnimationChannel &track : m_tracks)
    {
        duration = std::max(duration, std::visit([](const auto &typedTrack) {
                                return typedTrack.durationSeconds();
                            }, track));
    }
    return duration;
}

void AnimationClip::setDurationSeconds(float seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0f)
    {
        throw std::invalid_argument("Animation clip duration must be finite and non-negative");
    }
    m_durationSeconds = seconds;
}

} // namespace lr
