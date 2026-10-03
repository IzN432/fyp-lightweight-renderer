#pragma once

#include "core/scene/Component.hpp"
#include "features/animation/AnimationClip.hpp"

#include <cstddef>
#include <optional>
#include <variant>
#include <vector>

namespace lr
{

class TransformEditService;
class CommandManager;

class AnimatorComponent : public Component
{
public:
    struct KeyframeEdit
    {
        size_t                  clipIndex;
        size_t                  trackIndex;
        size_t                  keyframeIndex;
        SceneObjectId           target;
        AnimationTargetProperty property;
        float                   seconds;
    };

    explicit AnimatorComponent(std::vector<AnimationClip> clips = {});

    std::vector<AnimationClip> &clips() { return m_clips; }
    const std::vector<AnimationClip> &clips() const { return m_clips; }

    void play(size_t clipIndex);
    void pause() { m_playing = false; }
    void stop();
    void seek(float seconds);
    void update(float deltaSeconds);

    const std::optional<KeyframeEdit> &keyframeEdit() const { return m_keyframeEdit; }
    bool addKeyframe(size_t trackIndex, float seconds);
    bool beginKeyframeEdit(size_t trackIndex, size_t keyframeIndex);
    void applyKeyframeEdit();
    void cancelKeyframeEdit();

    bool isPlaying() const { return m_playing; }
    float playbackSeconds() const { return m_playbackSeconds; }
    std::optional<size_t> activeClipIndex() const { return m_activeClip; }

    bool loop() const { return m_loop; }
    void setLoop(bool loop) { m_loop = loop; }

    float speedMultiplier() const { return m_speedMultiplier; }
    void setSpeedMultiplier(float speedMultiplier);

    void onGUIImpl() override;

private:
    std::vector<AnimationClip> m_clips;
    std::optional<size_t>      m_activeClip;
    float                      m_playbackSeconds = 0.0f;
    float                      m_speedMultiplier = 1.0f;
    float                      m_trackViewCenter = 0.5f;
    float                      m_trackViewHalfWidth = 0.5f;
    size_t                     m_selectedTrack = 0;
    bool                       m_loop = true;
    bool                       m_playing = false;
    std::optional<KeyframeEdit> m_keyframeEdit;
    TransformEditService      *m_transformEditService = nullptr;
    CommandManager            *m_commandManager = nullptr;
};

} // namespace lr
