#include "core/editor/SceneObjectDragHandler.hpp"
#include "core/editor/SceneObjectRotationHandler.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"
#include "features/linear_blend_skinning/Skin.hpp"
#include "features/linear_blend_skinning/SkinStore.hpp"
#include "features/linear_blend_skinning/editor/SkeletonSelectionResolver.hpp"

#include <cassert>

namespace
{

lr::Joint joint(const lr::SceneObject &object)
{
    return {.sceneObject = object.id()};
}

bool near(const glm::vec3 &a, const glm::vec3 &b)
{
    return glm::length(a - b) < 1e-5f;
}

} // namespace

int main()
{
    lr::Scene scene;
    auto &root = scene.createSceneObject();
    auto &child = scene.createSceneObject();
    root.addComponent<lr::TransformComponent>(glm::vec3(2.0f, 0.0f, 0.0f));
    child.addComponent<lr::TransformComponent>(glm::vec3(0.0f, 1.0f, 0.0f));
    scene.setParent(child.id(), root.id());

    lr::SkinStore skins;
    lr::Skin skin(skins, skins.add(scene, {joint(root), joint(child)}));
    const lr::JointRef selectedJoint{skin.id(), 1};
    const std::optional<lr::SceneObjectId> targetId =
        lr::resolveSkeletonSelection(scene, skin, selectedJoint);
    assert(targetId == child.id());

    lr::CommandManager commands;
    lr::SceneObjectDragHandler handler(commands);
    handler.setTarget(&scene.getSceneObject(*targetId));

    bool committed = false;
    handler.setCommitCallback([&](lr::SceneObject &object) {
        assert(object.id() == child.id());
        // Auto Key is invoked from this callback. It must join the transform command's transaction
        // so one undo removes both the joint movement and its keyframe edit.
        assert(commands.inTransaction());
        committed = true;
    });

    handler.beginDrag();
    handler.translate(glm::vec3(1.0f, 0.0f, 0.0f));
    handler.endDrag(glm::vec3(1.0f, 0.0f, 0.0f));
    assert(committed);
    assert(near(child.getComponent<lr::TransformComponent>().transform().position(),
                glm::vec3(1.0f, 1.0f, 0.0f)));

    commands.undo();
    assert(near(child.getComponent<lr::TransformComponent>().transform().position(),
                glm::vec3(0.0f, 1.0f, 0.0f)));
    commands.redo();
    assert(near(child.getComponent<lr::TransformComponent>().transform().position(),
                glm::vec3(1.0f, 1.0f, 0.0f)));

    // Bone arrows are only hierarchy edges, not transforms. The endpoint popup resolves one to a
    // JointRef before the ordinary joint rotation handler is bound.
    const lr::BoneRef bone{skin.id(), 1};
    assert(!lr::resolveSkeletonSelection(scene, skin, bone));
    const lr::JointRef rootJoint{skin.id(), 0};
    const std::optional<lr::SceneObjectId> rotationTargetId =
        lr::resolveSkeletonSelection(scene, skin, rootJoint);
    assert(rotationTargetId == root.id());

    lr::SceneObjectRotationHandler rotationHandler(commands);
    rotationHandler.setTarget(&scene.getSceneObject(*rotationTargetId));
    bool rotationCommitted = false;
    rotationHandler.setCommitCallback([&](lr::SceneObject &object) {
        assert(object.id() == root.id());
        assert(commands.inTransaction());
        rotationCommitted = true;
    });

    const glm::vec3 childLocalBefore = child.getComponent<lr::TransformComponent>().transform().position();
    const float boneLengthBefore = glm::length(childLocalBefore);
    rotationHandler.beginDrag();
    const glm::quat quarterTurn = glm::angleAxis(glm::half_pi<float>(), glm::vec3(0.0f, 0.0f, 1.0f));
    const glm::mat4 rotatedRoot = glm::translate(glm::mat4(1.0f), glm::vec3(root.worldMatrix()[3])) *
                                  glm::mat4_cast(quarterTurn);
    rotationHandler.rotateToWorld(rotatedRoot);
    rotationHandler.endDrag();

    assert(rotationCommitted);
    assert(near(child.getComponent<lr::TransformComponent>().transform().position(), childLocalBefore));
    assert(glm::abs(glm::length(child.getComponent<lr::TransformComponent>().transform().position()) -
                    boneLengthBefore) < 1e-5f);
    assert(near(glm::vec3(child.worldMatrix()[3]), glm::vec3(1.0f, 1.0f, 0.0f)));

    commands.undo();
    assert(near(glm::vec3(child.worldMatrix()[3]), glm::vec3(3.0f, 1.0f, 0.0f)));
    commands.redo();
    assert(near(glm::vec3(child.worldMatrix()[3]), glm::vec3(1.0f, 1.0f, 0.0f)));
}
