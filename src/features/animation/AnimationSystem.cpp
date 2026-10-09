#include "AnimationSystem.hpp"
#include "core/scene/Scene.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace lr
{
std::string AnimationSystem::bindingKey(const AnimationTrackBinding &binding)
{
    return toString(binding.target) + ":" + std::to_string(static_cast<int>(binding.property));
}

AnimationPlayResult AnimationSystem::play(AnimationClipHandle handle, bool loop, float speed)
{
    if (!m_library.contains(handle)) throw std::out_of_range("Animation clip handle is out of range");
    if (!std::isfinite(speed) || speed < 0.0f) throw std::invalid_argument("Animation speed must be finite and non-negative");
    if (isPlaying(handle)) return {.started = true};

    std::unordered_set<std::string> requested;
    std::vector<AnimationTrackBinding> bindings;
    for (const AnimationChannel &channel : m_library.get(handle).tracks())
    {
        const AnimationTrackBinding binding = std::visit([](const auto &track) {
            return AnimationTrackBinding{track.target(), track.property()};
        }, channel);
        const std::string key = bindingKey(binding);
        if (!requested.insert(key).second)
            return {.conflict = binding, .conflictingClip = handle};
        if (const auto occupied = m_runningTracks.find(key); occupied != m_runningTracks.end())
            return {.conflict = binding, .conflictingClip = occupied->second};
        bindings.push_back(binding);
    }
    for (const auto &binding : bindings) m_runningTracks.emplace(bindingKey(binding), handle);
    m_playing.push_back({handle, 0.0f, speed, loop});
    return {.started = true};
}

bool AnimationSystem::isPlaying(AnimationClipHandle clip) const
{
    return std::ranges::any_of(m_playing, [clip](const Playback &item) { return item.clip == clip; });
}

std::optional<float> AnimationSystem::playbackTime(AnimationClipHandle clip) const
{
    const auto playback = std::ranges::find_if(m_playing, [clip](const Playback &item) { return item.clip == clip; });
    if (playback == m_playing.end()) return std::nullopt;
    return playback->seconds;
}

bool AnimationSystem::seek(AnimationClipHandle clip, float seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0f)
        throw std::invalid_argument("Animation seek time must be finite and non-negative");
    const auto playback = std::ranges::find_if(m_playing, [clip](const Playback &item) { return item.clip == clip; });
    if (playback == m_playing.end()) return false;

    const AnimationClip &animation = m_library.get(clip);
    playback->seconds = std::min(seconds, animation.durationSeconds());
    apply(animation, playback->seconds);
    return true;
}

void AnimationSystem::apply(const AnimationClip &clip, float seconds)
{
    for (const AnimationChannel &channel : clip.tracks())
        std::visit([&](const auto &track) { track.apply(m_scene, seconds); }, channel);
}

void AnimationSystem::release(AnimationClipHandle clip)
{
    std::erase_if(m_runningTracks, [clip](const auto &entry) { return entry.second == clip; });
}

void AnimationSystem::stop(AnimationClipHandle clip)
{
    std::erase_if(m_playing, [clip](const Playback &item) { return item.clip == clip; });
    release(clip);
}

void AnimationSystem::stopAll()
{
    m_playing.clear();
    m_runningTracks.clear();
}

void AnimationSystem::update(float deltaSeconds)
{
    if (!std::isfinite(deltaSeconds) || deltaSeconds < 0.0f)
        throw std::invalid_argument("Animation delta time must be finite and non-negative");
    std::vector<AnimationClipHandle> finished;
    for (Playback &playback : m_playing)
    {
        const AnimationClip &clip = m_library.get(playback.clip);
        const float duration = clip.durationSeconds();
        playback.seconds += deltaSeconds * playback.speed;
        if (duration <= 0.0f) { finished.push_back(playback.clip); continue; }
        if (playback.seconds >= duration)
        {
            if (playback.loop) playback.seconds = std::fmod(playback.seconds, duration);
            else { playback.seconds = duration; finished.push_back(playback.clip); }
        }
        apply(clip, playback.seconds);
    }
    for (AnimationClipHandle clip : finished) stop(clip);
}
} // namespace lr
