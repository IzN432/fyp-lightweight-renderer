#pragma once

#include "features/animation/AnimationTrack.hpp"

#include <string>
#include <vector>

namespace lr
{

class AnimationClip
{
public:
    explicit AnimationClip(std::string name = "Animation", std::vector<AnimationChannel> tracks = {});

    const std::string &name() const { return m_name; }
    void setName(std::string name);

    std::vector<AnimationChannel> &tracks() { return m_tracks; }
    const std::vector<AnimationChannel> &tracks() const { return m_tracks; }

    [[nodiscard]] float durationSeconds() const;

private:
    std::string                 m_name;
    std::vector<AnimationChannel> m_tracks;
};

} // namespace lr
