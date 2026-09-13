#include "features/animation/AnimationClip.hpp"

#include <algorithm>
#include <utility>

namespace lr
{

AnimationClip::AnimationClip(std::string name, std::vector<AnimationChannel> tracks)
    : m_name(std::move(name)), m_tracks(std::move(tracks))
{
    if (m_name.empty())
    {
        m_name = "Animation";
    }
}

void AnimationClip::setName(std::string name)
{
    m_name = name.empty() ? "Animation" : std::move(name);
}

float AnimationClip::durationSeconds() const
{
    float duration = 0.0f;
    for (const AnimationChannel &track : m_tracks)
    {
        duration = std::max(duration, std::visit([](const auto &typedTrack) {
                                return typedTrack.durationSeconds();
                            }, track));
    }
    return duration;
}

} // namespace lr
