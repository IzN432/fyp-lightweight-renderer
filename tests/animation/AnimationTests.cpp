#include "features/animation/AnimationClip.hpp"
#include "features/animation/AnimationSystem.hpp"
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
    // The object `track` drives has to exist before the track can name it. The tracks that are only
    // sampled, never applied, can name objects that were never created.
    lr::Scene scene;
    lr::AnimationLibrary animations;
    lr::AnimationSystem animationSystem(scene, animations);
    auto     &target          = scene.createSceneObject();
    auto     &targetTransform = target.addComponent<lr::TransformComponent>();

    const lr::SceneObjectId unusedTarget = lr::generateUuid();

    lr::TranslationTrack track(target.id());
    track.setKeyframe(2.0f, glm::vec3(4.0f, 0.0f, 0.0f));
    track.setKeyframe(0.0f, glm::vec3(0.0f));

    assert(track.target() == target.id());
    assert(near(track.durationSeconds(), 2.0f));
    assert(near(track.keyframes().front().seconds, 0.0f));
    assert(near(track.sample(1.0f)->x, 2.0f));
    assert(near(track.sample(-1.0f)->x, 0.0f));
    assert(near(track.sample(3.0f)->x, 4.0f));

    lr::ScaleTrack steppedTrack(unusedTarget, lr::AnimationInterpolation::Step);
    steppedTrack.setKeyframe(0.0f, glm::vec3(1.0f));
    steppedTrack.setKeyframe(2.0f, glm::vec3(3.0f));
    assert(near(steppedTrack.sample(1.0f)->x, 1.0f));

    lr::RotationTrack rotationTrack(unusedTarget);
    rotationTrack.setKeyframe(0.0f, glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    rotationTrack.setKeyframe(1.0f, glm::angleAxis(glm::pi<float>(), glm::vec3(0.0f, 1.0f, 0.0f)));
    assert(near(glm::length(rotationTrack.sample(0.5f).value()), 1.0f));

    lr::TranslationTrack cubicTrack(unusedTarget, lr::AnimationInterpolation::CubicSpline);
    cubicTrack.setKeyframe({.seconds = 0.0f, .value = glm::vec3(0.0f),
                            .outgoingTangent = glm::vec3(1.0f, 0.0f, 0.0f)});
    cubicTrack.setKeyframe({.seconds = 1.0f, .value = glm::vec3(1.0f, 0.0f, 0.0f),
                            .incomingTangent = glm::vec3(1.0f, 0.0f, 0.0f)});
    assert(near(cubicTrack.sample(0.5f)->x, 0.5f));

    lr::AnimationClip clip("Walk", {track});
    assert(near(clip.durationSeconds(), 2.0f));

    const lr::AnimationClipHandle clipHandle = animations.add(clip);
    lr::TranslationTrack conflictingTrack(target.id());
    conflictingTrack.setKeyframe(0.0f, glm::vec3(1.0f));
    const auto conflictingHandle = animations.add(
        lr::AnimationClip("Conflicting", {conflictingTrack}));
    lr::ScaleTrack independentTrack(target.id());
    independentTrack.setKeyframe(0.0f, glm::vec3(1.0f));
    const auto independentHandle = animations.add(
        lr::AnimationClip("Independent", {independentTrack}));
    assert(animationSystem.play(clipHandle, false, 2.0f).started);
    const auto conflict = animationSystem.play(conflictingHandle);
    assert(!conflict.started && conflict.conflictingClip == clipHandle);
    assert(animationSystem.play(independentHandle).started);
    animationSystem.update(0.5f);
    assert(near(targetTransform.transform().position().x, 2.0f));
    animationSystem.stopAll();
    assert(!animationSystem.isPlaying(clipHandle));

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
