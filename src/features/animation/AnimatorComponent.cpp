#include "features/animation/AnimatorComponent.hpp"

#include "core/app/ImGuiWidgets.hpp"
#include "core/editor/EditorContext.hpp"
#include "core/editor/command/Command.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"

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

namespace
{

class UpdateAnimationTrackCommand final : public Command
{
public:
    UpdateAnimationTrackCommand(AnimatorComponent &animator, size_t clipIndex, size_t trackIndex,
                                AnimationChannel before, AnimationChannel after, float playbackSeconds)
        : m_animator(animator), m_clipIndex(clipIndex), m_trackIndex(trackIndex),
          m_before(std::move(before)), m_after(std::move(after)), m_playbackSeconds(playbackSeconds)
    {}

    void execute() override { apply(m_after); }
    void undo() override { apply(m_before); }

private:
    void apply(const AnimationChannel &channel)
    {
        if (m_clipIndex >= m_animator.clips().size() ||
            m_trackIndex >= m_animator.clips()[m_clipIndex].tracks().size())
        {
            return;
        }
        m_animator.clips()[m_clipIndex].tracks()[m_trackIndex] = channel;
        m_animator.seek(m_playbackSeconds);
    }

    AnimatorComponent &m_animator;
    size_t              m_clipIndex;
    size_t              m_trackIndex;
    AnimationChannel    m_before;
    AnimationChannel    m_after;
    float               m_playbackSeconds;
};

} // namespace

AnimatorComponent::AnimatorComponent(std::vector<AnimationClip> clips)
    : Component("AnimatorComponent"), m_clips(std::move(clips))
{}

void AnimatorComponent::play(size_t clipIndex)
{
    if (clipIndex >= m_clips.size())
    {
        throw std::out_of_range("AnimatorComponent clip index is out of range");
    }
    cancelKeyframeEdit();
    m_activeClip = clipIndex;
    m_selectedTrack = 0;
    m_playbackSeconds = 0.0f;
    m_playing = true;
    markDirty();
}

void AnimatorComponent::stop()
{
    cancelKeyframeEdit();
    m_playing = false;
    m_activeClip.reset();
    m_playbackSeconds = 0.0f;
    markDirty();
}

bool AnimatorComponent::addKeyframe(size_t trackIndex, float seconds)
{
    if (!m_activeClip || m_keyframeEdit || !std::isfinite(seconds) || seconds < 0.0f ||
        trackIndex >= m_clips[*m_activeClip].tracks().size())
    {
        return false;
    }

    AnimationChannel &channel = m_clips[*m_activeClip].tracks()[trackIndex];
    const bool added = std::visit([&](auto &track) {
        Scene &scene = getOwningObject().scene();
        if (!scene.contains(track.target()) ||
            !scene.getSceneObject(track.target()).hasComponent<TransformComponent>())
        {
            return false;
        }

        const Transform &transform =
            scene.getSceneObject(track.target()).getComponent<TransformComponent>().transform();
        using Track = std::decay_t<decltype(track)>;
        if constexpr (std::is_same_v<Track, TranslationTrack>)
        {
            track.setKeyframe(seconds, transform.position());
        }
        else if constexpr (std::is_same_v<Track, RotationTrack>)
        {
            track.setKeyframe(seconds, transform.rotation());
        }
        else
        {
            track.setKeyframe(seconds, transform.scale());
        }
        return true;
    }, channel);

    if (added)
    {
        pause();
        seek(seconds);
        markDirty();
    }
    return added;
}

bool AnimatorComponent::deleteKeyframe(size_t trackIndex, size_t keyframeIndex)
{
    if (!m_activeClip || m_keyframeEdit || trackIndex >= m_clips[*m_activeClip].tracks().size())
    {
        return false;
    }

    AnimationChannel before = m_clips[*m_activeClip].tracks()[trackIndex];
    AnimationChannel after = before;
    const bool removed = std::visit([&](auto &track) {
        if (keyframeIndex >= track.keyframes().size())
        {
            return false;
        }
        return track.removeKeyframe(track.keyframes()[keyframeIndex].seconds);
    }, after);
    if (!removed)
    {
        return false;
    }

    pause();
    if (editorContext())
    {
        editorContext()->commands.executeCommand(std::make_unique<UpdateAnimationTrackCommand>(
            *this, *m_activeClip, trackIndex, std::move(before), std::move(after), m_playbackSeconds));
    }
    else
    {
        m_clips[*m_activeClip].tracks()[trackIndex] = std::move(after);
        seek(m_playbackSeconds);
    }
    markDirty();
    return true;
}

bool AnimatorComponent::beginKeyframeEdit(size_t trackIndex, size_t keyframeIndex)
{
    if (!m_activeClip || m_keyframeEdit || trackIndex >= m_clips[*m_activeClip].tracks().size())
    {
        return false;
    }

    AnimationChannel &channel = m_clips[*m_activeClip].tracks()[trackIndex];
    return std::visit([&](auto &track) {
        if (keyframeIndex >= track.keyframes().size())
        {
            return false;
        }
        Scene &scene = getOwningObject().scene();
        if (!scene.contains(track.target()) ||
            !scene.getSceneObject(track.target()).hasComponent<TransformComponent>())
        {
            return false;
        }

        pause();
        const float seconds = track.keyframes()[keyframeIndex].seconds;
        seek(seconds);
        m_keyframeEdit = KeyframeEdit{*m_activeClip, trackIndex, keyframeIndex, track.target(),
                                      std::decay_t<decltype(track)>::property(), seconds};
        if (editorContext())
        {
            m_commandManager = &editorContext()->commands;
            m_commandManager->beginTemporaryHistory();
            TransformTool tool = TransformTool::Scale;
            if constexpr (std::is_same_v<std::decay_t<decltype(track)>, TranslationTrack>)
            {
                tool = TransformTool::Translate;
            }
            else if constexpr (std::is_same_v<std::decay_t<decltype(track)>, RotationTrack>)
            {
                tool = TransformTool::Rotate;
            }
            m_transformEditService = &editorContext()->transformEdits;
            m_transformEditService->beginTransformEdit(scene.getSceneObject(track.target()), tool,
                                                       [this]() { cancelKeyframeEdit(); });
        }
        return true;
    }, channel);
}

void AnimatorComponent::applyKeyframeEdit()
{
    if (!m_keyframeEdit)
    {
        return;
    }

    const KeyframeEdit edit = *m_keyframeEdit;
    std::optional<AnimationChannel> before;
    std::optional<AnimationChannel> after;
    if (edit.clipIndex < m_clips.size() && edit.trackIndex < m_clips[edit.clipIndex].tracks().size())
    {
        AnimationChannel &channel = m_clips[edit.clipIndex].tracks()[edit.trackIndex];
        before = channel;
        Scene &scene = getOwningObject().scene();
        if (scene.contains(edit.target))
        {
            const Transform &transform =
                scene.getSceneObject(edit.target).getComponent<TransformComponent>().transform();
            std::visit([&](auto &track) {
                if (edit.keyframeIndex >= track.keyframes().size())
                {
                    return;
                }
                auto keyframe = track.keyframes()[edit.keyframeIndex];
                using Track = std::decay_t<decltype(track)>;
                if constexpr (std::is_same_v<Track, TranslationTrack>)
                {
                    keyframe.value = transform.position();
                }
                else if constexpr (std::is_same_v<Track, RotationTrack>)
                {
                    keyframe.value = transform.rotation();
                }
                else
                {
                    keyframe.value = transform.scale();
                }
                track.setKeyframe(std::move(keyframe));
            }, channel);
            after = channel;
        }
    }
    m_keyframeEdit.reset();
    if (m_transformEditService)
    {
        m_transformEditService->endTransformEdit();
        m_transformEditService = nullptr;
    }
    if (m_commandManager && before && after)
    {
        m_commandManager->replaceTemporaryHistory(std::make_unique<UpdateAnimationTrackCommand>(
            *this, edit.clipIndex, edit.trackIndex, std::move(*before), std::move(*after), edit.seconds));
    }
    else if (m_commandManager && m_commandManager->hasTemporaryHistory())
    {
        m_commandManager->cancelTemporaryHistory();
    }
    m_commandManager = nullptr;
    markDirty();
}

void AnimatorComponent::cancelKeyframeEdit()
{
    if (!m_keyframeEdit)
    {
        return;
    }
    const float seconds = m_keyframeEdit->seconds;
    m_keyframeEdit.reset();
    if (m_transformEditService)
    {
        m_transformEditService->endTransformEdit();
        m_transformEditService = nullptr;
    }
    if (m_commandManager && m_commandManager->hasTemporaryHistory())
    {
        m_commandManager->cancelTemporaryHistory();
    }
    m_commandManager = nullptr;
    seek(seconds);
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
                        objectName = target.name.empty() ? "Scene Object " + toString(track.target())
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
            ImGui::BeginDisabled(m_keyframeEdit.has_value());
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
            ImGui::EndDisabled();

            const AnimationChannel &selectedChannel = clip.tracks()[m_selectedTrack];
            enum class PendingKeyframeAction { None, Add, Edit, Delete, Apply, Cancel };
            PendingKeyframeAction pendingAction = PendingKeyframeAction::None;
            std::optional<size_t> pendingKeyframeIndex;
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
                        cancelKeyframeEdit();
                        pause();
                        seek(progress * duration);
                    }

                    std::optional<size_t> keyframeAtPlayhead;
                    for (size_t keyframeIndex = 0; keyframeIndex < track.keyframes().size(); ++keyframeIndex)
                    {
                        if (std::abs(track.keyframes()[keyframeIndex].seconds - m_playbackSeconds) <= 1e-4f)
                        {
                            keyframeAtPlayhead = keyframeIndex;
                            break;
                        }
                    }

                    const bool editingThisTrack = m_keyframeEdit &&
                        m_keyframeEdit->clipIndex == *m_activeClip &&
                        m_keyframeEdit->trackIndex == m_selectedTrack;

                    Scene &scene = getOwningObject().scene();
                    if (scene.contains(track.target()) &&
                        scene.getSceneObject(track.target()).hasComponent<TransformComponent>())
                    {
                        TransformComponent &transform =
                            scene.getSceneObject(track.target()).getComponent<TransformComponent>();
                        using Track = std::decay_t<decltype(track)>;
                        glm::vec3 value;
                        const char *valueLabel;
                        if constexpr (std::is_same_v<Track, TranslationTrack>)
                        {
                            value      = transform.transform().position();
                            valueLabel = "Position";
                        }
                        else if constexpr (std::is_same_v<Track, RotationTrack>)
                        {
                            value      = transform.transform().eulerDegrees();
                            valueLabel = "Rotation (degrees)";
                        }
                        else
                        {
                            value      = transform.transform().scale();
                            valueLabel = "Scale";
                        }

                        ImGui::BeginDisabled(!editingThisTrack);
                        if (ImGui::InputFloat3(valueLabel, &value.x, "%.6f"))
                        {
                            if constexpr (std::is_same_v<Track, TranslationTrack>)
                            {
                                transform.setPosition(value);
                            }
                            else if constexpr (std::is_same_v<Track, RotationTrack>)
                            {
                                transform.setEulerDegrees(value);
                            }
                            else
                            {
                                transform.setScale(value);
                            }
                        }
                        ImGui::EndDisabled();
                    }
                    else
                    {
                        ImGui::TextDisabled("Animated transform is unavailable");
                    }

                    if (editingThisTrack)
                    {
                        ImGui::Text("Editing keyframe at %.3f s", m_keyframeEdit->seconds);
                        if (ImGui::Button("Apply"))
                        {
                            pendingAction = PendingKeyframeAction::Apply;
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape))
                        {
                            pendingAction = PendingKeyframeAction::Cancel;
                        }
                    }
                    else
                    {
                        ImGui::BeginDisabled(keyframeAtPlayhead.has_value() || m_keyframeEdit.has_value());
                        if (ImGui::Button("Add Keyframe"))
                        {
                            pendingAction = PendingKeyframeAction::Add;
                        }
                        ImGui::EndDisabled();
                        ImGui::SameLine();
                        ImGui::BeginDisabled(!keyframeAtPlayhead.has_value() || m_keyframeEdit.has_value());
                        if (ImGui::Button("Edit Keyframe"))
                        {
                            pendingAction = PendingKeyframeAction::Edit;
                            pendingKeyframeIndex = keyframeAtPlayhead;
                        }
                        ImGui::EndDisabled();
                        ImGui::SameLine();
                        ImGui::BeginDisabled(!keyframeAtPlayhead.has_value() || m_keyframeEdit.has_value());
                        if (ImGui::Button("Delete Keyframe"))
                        {
                            pendingAction = PendingKeyframeAction::Delete;
                            pendingKeyframeIndex = keyframeAtPlayhead;
                        }
                        ImGui::EndDisabled();
                        if (!keyframeAtPlayhead)
                        {
                            ImGui::SameLine();
                            ImGui::TextDisabled("Captures the current transform");
                        }
                    }
                }, selectedChannel);
            switch (pendingAction)
            {
            case PendingKeyframeAction::Add:
                addKeyframe(m_selectedTrack, m_playbackSeconds);
                break;
            case PendingKeyframeAction::Edit:
                beginKeyframeEdit(m_selectedTrack, *pendingKeyframeIndex);
                break;
            case PendingKeyframeAction::Delete:
                deleteKeyframe(m_selectedTrack, *pendingKeyframeIndex);
                break;
            case PendingKeyframeAction::Apply:
                applyKeyframeEdit();
                break;
            case PendingKeyframeAction::Cancel:
                cancelKeyframeEdit();
                break;
            case PendingKeyframeAction::None:
                break;
            }
            ImGui::TreePop();
        }
    }
}

} // namespace lr
