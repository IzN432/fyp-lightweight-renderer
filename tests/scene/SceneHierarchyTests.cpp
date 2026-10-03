#include "core/editor/SceneObjectRotationHandler.hpp"
#include "core/editor/SceneObjectScaleHandler.hpp"
#include "core/loaders/MaterialStore.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/MeshStore.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cassert>
#include <stdexcept>
#include <vector>

int main()
{
    lr::Scene scene;
    auto     &root = scene.createSceneObject();
    auto     &child = scene.createSceneObject();
    auto     &grandchild = scene.createSceneObject();

    root.addComponent<lr::TransformComponent>().setPosition(glm::vec3(1.0f, 0.0f, 0.0f));
    child.addComponent<lr::TransformComponent>().setPosition(glm::vec3(0.0f, 2.0f, 0.0f));
    grandchild.addComponent<lr::TransformComponent>().setPosition(glm::vec3(0.0f, 0.0f, 3.0f));

    scene.setParent(child.id(), root.id());
    scene.setParent(grandchild.id(), child.id());

    assert(child.parent().value() == root.id());
    assert(root.children().front() == child.id());
    assert(glm::all(glm::epsilonEqual(glm::vec3(grandchild.worldMatrix()[3]),
                                     glm::vec3(1.0f, 2.0f, 3.0f), 0.0001f)));

    bool cycleRejected = false;
    try
    {
        scene.setParent(root.id(), grandchild.id());
    }
    catch (const std::invalid_argument &)
    {
        cycleRejected = true;
    }
    assert(cycleRejected);

    std::vector<lr::SceneObjectId> destroyed;
    scene.registerObjectsDestroyedCallback([&](std::span<const lr::SceneObjectId> ids) {
        destroyed.assign(ids.begin(), ids.end());
    });

    scene.destroySceneObject(child.id());
    assert(!scene.contains(child.id()));
    assert(!scene.contains(grandchild.id()));
    assert(root.children().empty());
    assert(destroyed.size() == 2);

    bool deletedLookupRejected = false;
    try
    {
        scene.getSceneObject(child.id());
    }
    catch (const std::out_of_range &)
    {
        deletedLookupRejected = true;
    }
    assert(deletedLookupRejected);

    auto &protectedObject = scene.createSceneObject();
    scene.protectSceneObject(protectedObject.id());
    assert(!scene.canDestroySceneObject(protectedObject.id()));
    bool protectedDeleteRejected = false;
    try
    {
        scene.destroySceneObject(protectedObject.id());
    }
    catch (const std::invalid_argument &)
    {
        protectedDeleteRejected = true;
    }
    assert(protectedDeleteRejected);
    assert(scene.contains(protectedObject.id()));

    lr::Mesh mesh;
    mesh.setTopology({glm::vec3(0.0f)}, {0}, {});
    lr::MeshStore meshStore;
    const lr::MeshHandle meshHandle = meshStore.add(std::move(mesh));
    lr::MaterialStore materialStore(1, [] { return lr::Material{}; });

    auto &firstInstance = scene.createSceneObject();
    auto &secondInstance = scene.createSceneObject();
    auto &firstMesh = firstInstance.addComponent<lr::MeshComponent>(meshHandle, meshStore,
                                                                    std::vector<lr::MaterialHandle>{}, materialStore);
    auto &secondMesh = secondInstance.addComponent<lr::MeshComponent>(meshHandle, meshStore,
                                                                      std::vector<lr::MaterialHandle>{}, materialStore);

    assert(firstMesh.meshHandle() == secondMesh.meshHandle());
    assert(&firstMesh.mesh() == &secondMesh.mesh());

    // A gizmo supplies a desired world matrix, but a child stores its rotation in parent-local
    // space. Verify conversion through a transformed parent and exact undo/redo restoration.
    auto &rotationParent = scene.createSceneObject();
    auto &rotationChild  = scene.createSceneObject();
    rotationParent.addComponent<lr::TransformComponent>(
        glm::vec3(2.0f, 1.0f, -3.0f), glm::angleAxis(glm::radians(25.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
        glm::vec3(2.0f, 3.0f, 4.0f));
    auto &childTransform = rotationChild.addComponent<lr::TransformComponent>(
        glm::vec3(1.0f, 2.0f, 3.0f), glm::angleAxis(glm::radians(10.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        glm::vec3(-1.5f, 0.75f, 2.0f));
    scene.setParent(rotationChild.id(), rotationParent.id());

    const glm::quat beforeRotation = childTransform.transform().rotation();
    const glm::quat desiredRotation = glm::normalize(
        glm::angleAxis(glm::radians(40.0f), glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f))));
    const glm::mat4 desiredLocal =
        glm::translate(glm::mat4(1.0f), childTransform.transform().position()) *
        glm::mat4_cast(desiredRotation) * glm::scale(glm::mat4(1.0f), childTransform.transform().scale());
    const glm::mat4 desiredWorld = rotationParent.worldMatrix() * desiredLocal;

    lr::CommandManager commandManager;
    lr::SceneObjectRotationHandler rotationHandler(commandManager);
    rotationHandler.setTarget(&rotationChild);
    rotationHandler.beginDrag();
    rotationHandler.rotateToWorld(desiredWorld);
    rotationHandler.endDrag();
    assert(glm::abs(glm::dot(childTransform.transform().rotation(), desiredRotation)) > 1.0f - 0.0001f);

    commandManager.undo();
    assert(glm::abs(glm::dot(childTransform.transform().rotation(), beforeRotation)) > 1.0f - 0.0001f);
    commandManager.redo();
    assert(glm::abs(glm::dot(childTransform.transform().rotation(), desiredRotation)) > 1.0f - 0.0001f);

    const glm::vec3 beforeScale = childTransform.transform().scale();
    const glm::vec3 desiredScale(-2.5f, 1.25f, 0.5f);
    const glm::mat4 desiredScaledLocal =
        glm::translate(glm::mat4(1.0f), childTransform.transform().position()) *
        glm::mat4_cast(childTransform.transform().rotation()) * glm::scale(glm::mat4(1.0f), desiredScale);
    const glm::mat4 desiredScaledWorld = rotationParent.worldMatrix() * desiredScaledLocal;

    lr::SceneObjectScaleHandler scaleHandler(commandManager);
    scaleHandler.setTarget(&rotationChild);
    scaleHandler.beginDrag();
    scaleHandler.scaleToWorld(desiredScaledWorld);
    scaleHandler.endDrag();
    assert(glm::all(glm::epsilonEqual(childTransform.transform().scale(), desiredScale, 0.0001f)));

    commandManager.undo();
    assert(glm::all(glm::epsilonEqual(childTransform.transform().scale(), beforeScale, 0.0001f)));
    commandManager.redo();
    assert(glm::all(glm::epsilonEqual(childTransform.transform().scale(), desiredScale, 0.0001f)));
}
