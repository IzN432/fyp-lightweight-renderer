#include "core/editor/SceneObjectRotationHandler.hpp"
#include "core/editor/SceneObjectScaleHandler.hpp"
#include "core/editor/command/Command.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/loaders/MaterialStore.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/MeshStore.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{

class SetValueCommand final : public lr::Command
{
public:
    SetValueCommand(int &value, int after) : m_value(value), m_before(value), m_after(after) {}
    SetValueCommand(int &value, int before, int after) : m_value(value), m_before(before), m_after(after) {}
    void execute() override { m_value = m_after; }
    void undo() override { m_value = m_before; }

private:
    int &m_value;
    int  m_before;
    int  m_after;
};

} // namespace

int main()
{
    int commandValue = 0;
    lr::CommandManager scopedCommands;
    scopedCommands.executeCommand(std::make_unique<SetValueCommand>(commandValue, 1));
    scopedCommands.beginTemporaryHistory();
    scopedCommands.executeCommand(std::make_unique<SetValueCommand>(commandValue, 2));
    scopedCommands.cancelTemporaryHistory();
    assert(commandValue == 1);

    scopedCommands.beginTemporaryHistory();
    scopedCommands.executeCommand(std::make_unique<SetValueCommand>(commandValue, 3));
    scopedCommands.replaceTemporaryHistory(std::make_unique<SetValueCommand>(commandValue, 1, 4));
    assert(commandValue == 4);
    scopedCommands.undo();
    assert(commandValue == 1);
    scopedCommands.redo();
    assert(commandValue == 4);

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
    auto destroyedConnection = scene.registerObjectsDestroyedCallback([&](std::span<const lr::SceneObjectId> ids) {
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
    const glm::quat desiredWorldRotation = glm::normalize(rotationParent.worldRotation() * desiredRotation);
    const glm::mat4 desiredWorld =
        glm::translate(glm::mat4(1.0f), glm::vec3((rotationParent.worldMatrix() * desiredLocal)[3])) *
        glm::mat4_cast(desiredWorldRotation);

    // The full render matrix is sheared by the parent's non-uniform scale, while the rigid frame
    // supplied to the rotation gizmo remains orthonormal so its rendered rings and hit planes agree.
    const glm::mat3 shearedWorld(rotationChild.worldMatrix());
    const float shearAmount =
        glm::max(glm::abs(glm::dot(glm::normalize(shearedWorld[0]), glm::normalize(shearedWorld[1]))),
                 glm::max(glm::abs(glm::dot(glm::normalize(shearedWorld[0]), glm::normalize(shearedWorld[2]))),
                          glm::abs(glm::dot(glm::normalize(shearedWorld[1]), glm::normalize(shearedWorld[2])))));
    assert(shearAmount > 0.0001f);
    const glm::mat3 rigidWorld(glm::mat4_cast(rotationChild.worldRotation()));
    assert(glm::abs(glm::dot(rigidWorld[0], rigidWorld[1])) < 0.0001f);
    assert(glm::abs(glm::dot(rigidWorld[0], rigidWorld[2])) < 0.0001f);
    assert(glm::abs(glm::dot(rigidWorld[1], rigidWorld[2])) < 0.0001f);

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

    // Temporary edits (such as animation keyframe editing) still apply the gizmo result while
    // suppressing creation of a separate undo command.
    const glm::quat temporaryRotation =
        glm::angleAxis(glm::radians(65.0f), glm::normalize(glm::vec3(0.0f, 1.0f, 1.0f)));
    const glm::quat temporaryWorldRotation =
        glm::normalize(rotationParent.worldRotation() * temporaryRotation);
    const glm::mat4 temporaryWorld =
        glm::translate(glm::mat4(1.0f), glm::vec3(rotationChild.worldMatrix()[3])) *
        glm::mat4_cast(temporaryWorldRotation);
    rotationHandler.setRecordCommands(false);
    rotationHandler.beginDrag();
    rotationHandler.rotateToWorld(temporaryWorld);
    rotationHandler.endDrag();
    rotationHandler.setRecordCommands(true);
    assert(glm::abs(glm::dot(childTransform.transform().rotation(), temporaryRotation)) > 1.0f - 0.0001f);

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
