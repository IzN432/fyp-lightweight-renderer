#include "features/animation/AnimationClip.hpp"
#include "features/animation/AnimatorComponent.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace
{

bool near(float lhs, float rhs)
{
    return std::abs(lhs - rhs) < 0.0001f;
}

} // namespace

int main()
{
    lr::TranslationTrack track(0);
    track.setKeyframe(2.0f, glm::vec3(4.0f, 0.0f, 0.0f));
    track.setKeyframe(0.0f, glm::vec3(0.0f));

    assert(track.target() == 0);
    assert(near(track.durationSeconds(), 2.0f));
    assert(near(track.keyframes().front().seconds, 0.0f));
    assert(near(track.sample(1.0f)->x, 2.0f));
    assert(near(track.sample(-1.0f)->x, 0.0f));
    assert(near(track.sample(3.0f)->x, 4.0f));

    lr::ScaleTrack steppedTrack(8, lr::AnimationInterpolation::Step);
    steppedTrack.setKeyframe(0.0f, glm::vec3(1.0f));
    steppedTrack.setKeyframe(2.0f, glm::vec3(3.0f));
    assert(near(steppedTrack.sample(1.0f)->x, 1.0f));

    lr::RotationTrack rotationTrack(9);
    rotationTrack.setKeyframe(0.0f, glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    rotationTrack.setKeyframe(1.0f, glm::angleAxis(glm::pi<float>(), glm::vec3(0.0f, 1.0f, 0.0f)));
    assert(near(glm::length(rotationTrack.sample(0.5f).value()), 1.0f));

    lr::TranslationTrack cubicTrack(10, lr::AnimationInterpolation::CubicSpline);
    cubicTrack.setKeyframe({.seconds = 0.0f, .value = glm::vec3(0.0f),
                            .outgoingTangent = glm::vec3(1.0f, 0.0f, 0.0f)});
    cubicTrack.setKeyframe({.seconds = 1.0f, .value = glm::vec3(1.0f, 0.0f, 0.0f),
                            .incomingTangent = glm::vec3(1.0f, 0.0f, 0.0f)});
    assert(near(cubicTrack.sample(0.5f)->x, 0.5f));

    lr::AnimationClip clip("Walk", {track});
    assert(near(clip.durationSeconds(), 2.0f));

    lr::Scene scene;
    auto &target = scene.createSceneObject();
    auto &targetTransform = target.addComponent<lr::TransformComponent>();
    auto &animatorObject = scene.createSceneObject();
    auto &animator = animatorObject.addComponent<lr::AnimatorComponent>(std::vector<lr::AnimationClip>{clip});
    animator.setLoop(false);
    animator.setSpeedMultiplier(2.0f);
    animator.play(0);
    animator.update(0.5f);
    assert(near(animator.playbackSeconds(), 1.0f));
    assert(near(targetTransform.transform().position().x, 2.0f));
    animator.update(1.0f);
    assert(!animator.isPlaying());
    assert(near(animator.playbackSeconds(), 2.0f));

    assert(animator.beginKeyframeEdit(0, 1));
    targetTransform.setPosition(glm::vec3(9.0f, 0.0f, 0.0f));
    animator.cancelKeyframeEdit();
    assert(near(targetTransform.transform().position().x, 4.0f));
    assert(near(std::get<lr::TranslationTrack>(animator.clips()[0].tracks()[0]).sample(2.0f)->x, 4.0f));

    assert(animator.beginKeyframeEdit(0, 1));
    targetTransform.setPosition(glm::vec3(7.0f, 0.0f, 0.0f));
    animator.applyKeyframeEdit();
    assert(!animator.keyframeEdit());
    assert(near(std::get<lr::TranslationTrack>(animator.clips()[0].tracks()[0]).sample(2.0f)->x, 7.0f));

    bool rejectedInvalidTime = false;
    try
    {
        track.setKeyframe(-1.0f, glm::vec3(0.0f));
    }
    catch (const std::invalid_argument &)
    {
        rejectedInvalidTime = true;
    }
    assert(rejectedInvalidTime);
}
