#pragma once

#include "features/animation/AnimationLibrary.hpp"
#include "features/animation/AnimationTrack.hpp"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lr
{
class Scene;

struct AnimationTrackBinding
{
    SceneObjectId target;
    AnimationTargetProperty property;
};

struct AnimationPlayResult
{
    bool started = false;
    std::optional<AnimationTrackBinding> conflict;
    std::optional<AnimationClipHandle> conflictingClip;
};

class AnimationSystem
{
public:
    AnimationSystem(Scene &scene, AnimationLibrary &library) : m_scene(scene), m_library(library) {}

    AnimationPlayResult play(AnimationClipHandle clip, bool loop = true, float speed = 1.0f);
    void stop(AnimationClipHandle clip);
    // Removes a clip and compacts the library while preserving the meaning of every live runtime
    // handle above it.
    void remove(AnimationClipHandle clip);
    void removeTrack(AnimationClipHandle clip, size_t track);
    void stopAll();
    void update(float deltaSeconds);
    bool isPlaying(AnimationClipHandle clip) const;
    std::optional<float> playbackTime(AnimationClipHandle clip) const;
    bool seek(AnimationClipHandle clip, float seconds);

private:
    struct Playback { AnimationClipHandle clip; float seconds; float speed; bool loop; };
    static std::string bindingKey(const AnimationTrackBinding &binding);
    void apply(const AnimationClip &clip, float seconds);
    void release(AnimationClipHandle clip);

    Scene &m_scene;
    AnimationLibrary &m_library;
    std::vector<Playback> m_playing;
    std::unordered_map<std::string, AnimationClipHandle> m_runningTracks;
};
} // namespace lr
