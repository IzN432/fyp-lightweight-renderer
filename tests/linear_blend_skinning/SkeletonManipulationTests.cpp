#include "core/editor/SceneObjectDragHandler.hpp"
#include "core/editor/SceneObjectRotationHandler.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"
#include "features/linear_blend_skinning/Skin.hpp"
#include "features/linear_blend_skinning/SkinStore.hpp"
#include "features/linear_blend_skinning/editor/SkeletonManipulationAdapter.hpp"

#include <cassert>
#include <glm/gtc/constants.hpp>

namespace
{
lr::Joint joint(const lr::SceneObject &object) { return {.sceneObject = object.id()}; }
bool near(const glm::vec3 &a, const glm::vec3 &b) { return glm::length(a - b) < 1e-5f; }

void adapterSemanticsAndSharedSkins()
{
    lr::Scene scene;
    auto &root = scene.createSceneObject();
    auto &child = scene.createSceneObject();
    root.addComponent<lr::TransformComponent>();
    child.addComponent<lr::TransformComponent>();
    scene.setParent(child.id(), root.id());
    lr::SkinStore skins;
    const lr::SkinHandle handle = skins.add(scene, {joint(root), joint(child)});
    lr::Skin firstMeshSkin(skins, handle), secondMeshSkin(skins, handle);

    const auto jointTarget = lr::SkeletonManipulationAdapter::resolve(
        scene, firstMeshSkin, lr::JointRef{firstMeshSkin.id(), 0});
    assert(jointTarget && jointTarget->object == root.id() && jointTarget->joint == 0);
    assert(!jointTarget->selectedThroughBone);

    static_assert(lr::SkeletonManipulationAdapter::boneSemantics ==
                  lr::BoneShaftPoseSemantics::ChildJointLocalTransform);
    const lr::BoneRef shaft{firstMeshSkin.id(), 1};
    const auto firstTarget = lr::SkeletonManipulationAdapter::resolve(scene, firstMeshSkin, shaft);
    const auto secondTarget = lr::SkeletonManipulationAdapter::resolve(scene, secondMeshSkin, shaft);
    assert(firstTarget && secondTarget);
    assert(firstTarget->object == child.id() && secondTarget->object == child.id());
    assert(firstTarget->selectedThroughBone && secondTarget->selectedThroughBone);

    lr::Skin otherSkin(skins, skins.add(scene, {joint(root)}));
    assert(!lr::SkeletonManipulationAdapter::resolve(scene, otherSkin, shaft));
    assert(!lr::SkeletonManipulationAdapter::resolve(
        scene, firstMeshSkin, lr::JointRef{firstMeshSkin.id(), 99}));
}

void translationWithRotatedAndScaledParent()
{
    lr::Scene scene;
    auto &parent = scene.createSceneObject();
    auto &child = scene.createSceneObject();
    const glm::quat quarterTurn = glm::angleAxis(glm::half_pi<float>(), glm::vec3(0, 0, 1));
    parent.addComponent<lr::TransformComponent>(glm::vec3(2, 0, 0), quarterTurn, glm::vec3(2, 4, 1));
    child.addComponent<lr::TransformComponent>(glm::vec3(1, 1, 0));
    scene.setParent(child.id(), parent.id());

    lr::CommandManager commands;
    lr::SceneObjectDragHandler handler(commands);
    handler.setTarget(&child);
    bool committed = false;
    handler.setCommitCallback([&](lr::SceneObject &) { assert(commands.inTransaction()); committed = true; });
    const glm::vec3 worldBefore(child.worldMatrix()[3]);
    handler.beginDrag();
    handler.translate(glm::vec3(4, 0, 0));
    handler.endDrag(glm::vec3(4, 0, 0));
    assert(committed);
    assert(near(glm::vec3(child.worldMatrix()[3]), worldBefore + glm::vec3(4, 0, 0)));
    assert(near(child.getComponent<lr::TransformComponent>().transform().position(), glm::vec3(1, 0, 0)));
    commands.undo();
    assert(near(child.getComponent<lr::TransformComponent>().transform().position(), glm::vec3(1, 1, 0)));
    commands.redo();
    assert(near(glm::vec3(child.worldMatrix()[3]), worldBefore + glm::vec3(4, 0, 0)));
}

void rootTranslationAndChildRotation()
{
    lr::Scene scene;
    auto &root = scene.createSceneObject();
    auto &child = scene.createSceneObject();
    root.addComponent<lr::TransformComponent>(glm::vec3(2, 0, 0));
    child.addComponent<lr::TransformComponent>(glm::vec3(0, 1, 0));
    scene.setParent(child.id(), root.id());
    lr::CommandManager commands;
    lr::SceneObjectDragHandler translation(commands);
    translation.setTarget(&root);
    translation.beginDrag();
    translation.translate(glm::vec3(1, 2, 0));
    translation.endDrag(glm::vec3(1, 2, 0));
    assert(near(root.getComponent<lr::TransformComponent>().transform().position(), glm::vec3(3, 2, 0)));
    commands.undo();

    lr::SceneObjectRotationHandler rotation(commands);
    rotation.setTarget(&child);
    const glm::vec3 childPosition = child.getComponent<lr::TransformComponent>().transform().position();
    const glm::quat quarterTurn = glm::angleAxis(glm::half_pi<float>(), glm::vec3(0, 0, 1));
    rotation.beginDrag();
    rotation.rotateToWorld(glm::translate(glm::mat4(1), glm::vec3(child.worldMatrix()[3])) *
                           glm::mat4_cast(quarterTurn));
    rotation.endDrag();
    assert(near(child.getComponent<lr::TransformComponent>().transform().position(), childPosition));
    assert(near(child.worldRotation() * glm::vec3(1, 0, 0), glm::vec3(0, 1, 0)));
    commands.undo();
    assert(near(child.worldRotation() * glm::vec3(1, 0, 0), glm::vec3(1, 0, 0)));
    commands.redo();
    assert(near(child.worldRotation() * glm::vec3(1, 0, 0), glm::vec3(0, 1, 0)));
}

void rotationUnderRotatedScaledParent()
{
    lr::Scene scene;
    auto &parent = scene.createSceneObject();
    auto &child = scene.createSceneObject();
    const glm::quat parentRotation = glm::angleAxis(glm::half_pi<float>(), glm::vec3(0, 1, 0));
    parent.addComponent<lr::TransformComponent>(glm::vec3(0), parentRotation, glm::vec3(2, 3, 4));
    child.addComponent<lr::TransformComponent>();
    scene.setParent(child.id(), parent.id());
    lr::CommandManager commands;
    lr::SceneObjectRotationHandler rotation(commands);
    rotation.setTarget(&child);
    const glm::quat desiredWorld = glm::angleAxis(glm::half_pi<float>(), glm::vec3(1, 0, 0));
    rotation.beginDrag();
    rotation.rotateToWorld(glm::mat4_cast(desiredWorld));
    rotation.endDrag();
    assert(glm::abs(glm::dot(child.worldRotation(), desiredWorld)) > 1.0f - 1e-5f);
    assert(glm::abs(glm::dot(child.getComponent<lr::TransformComponent>().transform().rotation(),
                             glm::normalize(glm::inverse(parentRotation) * desiredWorld))) > 1.0f - 1e-5f);
}
} // namespace

int main()
{
    adapterSemanticsAndSharedSkins();
    translationWithRotatedAndScaledParent();
    rootTranslationAndChildRotation();
    rotationUnderRotatedScaledParent();
}
