#pragma once

#include "features/animation/AnimationTrack.hpp"

#include <string>
#include <optional>
#include <vector>

namespace lr
{

class AnimationClip
{
public:
    explicit AnimationClip(std::string name = "Animation", std::vector<AnimationChannel> tracks = {},
                           std::optional<float> durationSeconds = std::nullopt);

    const std::string &name() const { return m_name; }
    void setName(std::string name);

    std::vector<AnimationChannel> &tracks() { return m_tracks; }
    const std::vector<AnimationChannel> &tracks() const { return m_tracks; }

    [[nodiscard]] float durationSeconds() const;
    void setDurationSeconds(float seconds);

private:
    std::string                 m_name;
    std::vector<AnimationChannel> m_tracks;
    // Missing only for legacy/imported clips whose duration has not been authored explicitly. In
    // that case durationSeconds() retains the old last-keyframe behavior until the editor sets it.
    std::optional<float>        m_durationSeconds;
};

} // namespace lr
