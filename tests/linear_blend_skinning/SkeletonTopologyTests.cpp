#include "core/scene/Scene.hpp"
#include "features/linear_blend_skinning/Skin.hpp"
#include "features/linear_blend_skinning/SkinComponent.hpp"
#include "features/linear_blend_skinning/editor/SkeletonTopology.hpp"

#include <cassert>
#include <stdexcept>

namespace
{

lr::Joint joint(const lr::SceneObject &object)
{
    return {.sceneObject = object.id()};
}

} // namespace

int main()
{
    lr::Scene scene;
    auto &skinObject = scene.createSceneObject();
    auto &outsideParent = scene.createSceneObject();
    auto &root = scene.createSceneObject();
    auto &middle = scene.createSceneObject();
    auto &leafA = scene.createSceneObject();
    auto &leafB = scene.createSceneObject();

    scene.setParent(root.id(), outsideParent.id());
    scene.setParent(middle.id(), root.id());
    scene.setParent(leafA.id(), middle.id());
    scene.setParent(leafB.id(), middle.id());

    // Deliberately not hierarchy order: palette indices, not traversal order, identify bones.
    lr::Skin skin(scene, {joint(leafA), joint(root), joint(leafB), joint(middle)});
    lr::SkeletonTopology topology(skinObject.id(), scene, skin);

    assert(topology.bonesByChildJoint().size() == 3);
    assert(!topology.bonesByChildJoint().contains(1)); // outsideParent is not part of this skin.
    assert(topology.bonesByChildJoint().at(0).parentJoint == 3);
    assert(topology.bonesByChildJoint().at(2).parentJoint == 3);
    assert(topology.bonesByChildJoint().at(3).parentJoint == 1);

    const lr::BoneRef leafBone{skinObject.id(), 0};
    assert(topology.find(leafBone));
    assert(topology.find(leafBone)->childObject == leafA.id());

    // Joint balls use the Skin palette directly; topology stores only actual connections.
    const lr::JointRef middleJoint{skinObject.id(), 3};
    const lr::JointRef rootJoint{skinObject.id(), 1};
    assert(skin.joints()[middleJoint.joint].sceneObject == middle.id());
    assert(skin.joints()[rootJoint.joint].sceneObject == root.id());

    // A root has a selectable joint ball but no incoming bone shaft.
    assert(!topology.find(lr::BoneRef{skinObject.id(), 1}));
    assert(!topology.find(lr::BoneRef{lr::generateUuid(), 0}));
    assert(!topology.find(lr::BoneRef{skinObject.id(), 99}));

    bool duplicateRejected = false;
    try
    {
        lr::Skin duplicateSkin(scene, {joint(root), joint(root)});
        lr::SkeletonTopology duplicateTopology(skinObject.id(), scene, duplicateSkin);
    }
    catch (const std::invalid_argument &)
    {
        duplicateRejected = true;
    }
    assert(duplicateRejected);

    // A skin outside the joint subtree prevents deletion from leaving a dangling palette entry.
    skinObject.addComponent<lr::SkinComponent>(lr::Skin(scene, {joint(root), joint(middle)}));
    const lr::Scene::DestructionCheck destruction = scene.checkSceneObjectDestruction(root.id());
    assert(!destruction.allowed);
    assert(destruction.reason.find("SkinComponent") != std::string::npos);
    assert(!scene.canDestroySceneObject(root.id()));
    bool referencedDeleteRejected = false;
    try
    {
        scene.destroySceneObject(root.id());
    }
    catch (const std::invalid_argument &)
    {
        referencedDeleteRejected = true;
    }
    assert(referencedDeleteRejected);
    assert(scene.contains(root.id()));

    // A component being deleted with its referenced joints cannot leave a dangling reference.
    lr::Scene ownedRigScene;
    auto &ownedSkin = ownedRigScene.createSceneObject();
    auto &ownedJoint = ownedRigScene.createSceneObject();
    ownedRigScene.setParent(ownedJoint.id(), ownedSkin.id());
    ownedSkin.addComponent<lr::SkinComponent>(lr::Skin(ownedRigScene, {joint(ownedJoint)}));
    assert(ownedRigScene.canDestroySceneObject(ownedSkin.id()));
    ownedRigScene.destroySceneObject(ownedSkin.id());
    assert(!ownedRigScene.contains(ownedSkin.id()));
    assert(!ownedRigScene.contains(ownedJoint.id()));
}
