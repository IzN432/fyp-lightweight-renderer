#include "features/animation/AnimationTrack.hpp"

#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <glm/common.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace lr
{

template <typename T, AnimationTargetProperty Property>
AnimationTrack<T, Property>::AnimationTrack(SceneObjectId target, AnimationInterpolation interpolation)
    : m_target(target), m_interpolation(interpolation)
{}

template <typename T, AnimationTargetProperty Property>
void AnimationTrack<T, Property>::setKeyframe(Keyframe keyframe)
{
    if (!std::isfinite(keyframe.seconds) || keyframe.seconds < 0.0f)
    {
        throw std::invalid_argument("AnimationTrack keyframe time must be finite and non-negative");
    }
    const auto position = std::lower_bound(m_keyframes.begin(), m_keyframes.end(), keyframe.seconds,
                                           [](const Keyframe &candidate, float seconds) {
                                               return candidate.seconds < seconds;
                                           });
    if (position != m_keyframes.end() && position->seconds == keyframe.seconds)
    {
        *position = std::move(keyframe);
    }
    else
    {
        m_keyframes.insert(position, std::move(keyframe));
    }
}

template <typename T, AnimationTargetProperty Property>
void AnimationTrack<T, Property>::setKeyframe(float seconds, T value)
{
    setKeyframe({.seconds = seconds, .value = std::move(value)});
}

template <typename T, AnimationTargetProperty Property>
bool AnimationTrack<T, Property>::removeKeyframe(float seconds)
{
    const auto position = std::lower_bound(m_keyframes.begin(), m_keyframes.end(), seconds,
                                           [](const Keyframe &candidate, float candidateSeconds) {
                                               return candidate.seconds < candidateSeconds;
                                           });
    if (position == m_keyframes.end() || position->seconds != seconds)
    {
        return false;
    }
    m_keyframes.erase(position);
    return true;
}

template <typename T, AnimationTargetProperty Property>
float AnimationTrack<T, Property>::durationSeconds() const
{
    return m_keyframes.empty() ? 0.0f : m_keyframes.back().seconds;
}

template <typename T, AnimationTargetProperty Property>
std::optional<T> AnimationTrack<T, Property>::sample(float seconds) const
{
    if (m_keyframes.empty())
    {
        return std::nullopt;
    }
    const auto upper = std::upper_bound(m_keyframes.begin(), m_keyframes.end(), seconds,
                                        [](float candidateSeconds, const Keyframe &candidate) {
                                            return candidateSeconds < candidate.seconds;
                                        });
    if (upper == m_keyframes.begin())
    {
        return upper->value;
    }
    if (upper == m_keyframes.end())
    {
        return m_keyframes.back().value;
    }

    const Keyframe &lower = *std::prev(upper);
    if (m_interpolation == AnimationInterpolation::Step)
    {
        return lower.value;
    }

    const float span = upper->seconds - lower.seconds;
    const float alpha = span > 0.0f ? glm::clamp((seconds - lower.seconds) / span, 0.0f, 1.0f) : 0.0f;
    if (m_interpolation == AnimationInterpolation::CubicSpline)
    {
        const float alpha2 = alpha * alpha;
        const float alpha3 = alpha2 * alpha;
        T result = (2.0f * alpha3 - 3.0f * alpha2 + 1.0f) * lower.value +
                   (alpha3 - 2.0f * alpha2 + alpha) * span * lower.outgoingTangent +
                   (-2.0f * alpha3 + 3.0f * alpha2) * upper->value +
                   (alpha3 - alpha2) * span * upper->incomingTangent;
        if constexpr (std::is_same_v<T, glm::quat>)
        {
            result = glm::normalize(result);
        }
        return result;
    }

    if constexpr (std::is_same_v<T, glm::quat>)
    {
        return glm::normalize(glm::slerp(lower.value, upper->value, alpha));
    }
    else
    {
        return glm::mix(lower.value, upper->value, alpha);
    }
}

template <typename T, AnimationTargetProperty Property>
void AnimationTrack<T, Property>::apply(Scene &scene, float seconds) const
{
    if (!scene.contains(m_target))
    {
        return;
    }
    const std::optional<T> value = sample(seconds);
    if (!value)
    {
        return;
    }

    TransformComponent &transform = scene.getSceneObject(m_target).getComponent<TransformComponent>();
    if constexpr (Property == AnimationTargetProperty::Translation)
    {
        transform.setPosition(value.value());
    }
    else if constexpr (Property == AnimationTargetProperty::Rotation)
    {
        transform.setRotation(value.value());
    }
    else
    {
        transform.setScale(value.value());
    }
}

template class AnimationTrack<glm::vec3, AnimationTargetProperty::Translation>;
template class AnimationTrack<glm::quat, AnimationTargetProperty::Rotation>;
template class AnimationTrack<glm::vec3, AnimationTargetProperty::Scale>;

} // namespace lr
