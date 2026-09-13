#pragma once

#include "core/scene/SceneObjectId.hpp"

#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

#include <optional>
#include <variant>
#include <vector>

namespace lr
{

class Scene;

enum class AnimationInterpolation
{
    Linear,
    Step,
    CubicSpline
};

enum class AnimationTargetProperty
{
    Translation,
    Rotation,
    Scale
};

template <typename T> struct AnimationKeyframe
{
    float seconds = 0.0f;
    T value{};
    T incomingTangent{};
    T outgoingTangent{};
};

template <typename T, AnimationTargetProperty Property> class AnimationTrack
{
public:
    using Value = T;
    using Keyframe = AnimationKeyframe<T>;

    explicit AnimationTrack(SceneObjectId target,
                            AnimationInterpolation interpolation = AnimationInterpolation::Linear);

    SceneObjectId target() const { return m_target; }
    static constexpr AnimationTargetProperty property() { return Property; }

    AnimationInterpolation interpolation() const { return m_interpolation; }
    void setInterpolation(AnimationInterpolation interpolation) { m_interpolation = interpolation; }

    const std::vector<Keyframe> &keyframes() const { return m_keyframes; }

    void setKeyframe(Keyframe keyframe);
    void setKeyframe(float seconds, T value);
    bool removeKeyframe(float seconds);

    [[nodiscard]] float durationSeconds() const;
    [[nodiscard]] std::optional<T> sample(float seconds) const;
    void apply(Scene &scene, float seconds) const;

private:
    SceneObjectId          m_target;
    AnimationInterpolation m_interpolation;
    std::vector<Keyframe>  m_keyframes;
};

using TranslationTrack = AnimationTrack<glm::vec3, AnimationTargetProperty::Translation>;
using RotationTrack = AnimationTrack<glm::quat, AnimationTargetProperty::Rotation>;
using ScaleTrack = AnimationTrack<glm::vec3, AnimationTargetProperty::Scale>;
using AnimationChannel = std::variant<TranslationTrack, RotationTrack, ScaleTrack>;

extern template class AnimationTrack<glm::vec3, AnimationTargetProperty::Translation>;
extern template class AnimationTrack<glm::quat, AnimationTargetProperty::Rotation>;
extern template class AnimationTrack<glm::vec3, AnimationTargetProperty::Scale>;

} // namespace lr
