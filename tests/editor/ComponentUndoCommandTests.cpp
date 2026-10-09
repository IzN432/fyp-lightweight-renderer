#include "core/editor/command/AddComponentCommand.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/editor/command/EditComponentValuesCommand.hpp"
#include "core/editor/command/RemoveComponentCommand.hpp"
#include "core/editor/command/RenameSceneObjectCommand.hpp"
#include "core/editor/EditorContext.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <cassert>
#include <memory>
#include <typeindex>
#include <variant>

namespace
{

// Counts the notifications the editor uses to bring its GPU-side state up to date, so the tests can
// check that putting a component back announces itself the same way adding one does.
class CountingAddService final : public lr::ComponentAddService
{
public:
    void onComponentAdded(lr::SceneObject &) override { ++additions; }

    int additions = 0;
};

float intensityOf(const lr::Light &light)
{
    return std::visit(
        [](const auto &value) {
            return static_cast<const lr::BaseLight &>(value).intensity;
        },
        light.light);
}

const std::type_index lightType(typeid(lr::Light));

} // namespace

int main()
{
    // An edit of a component's values is restored and reapplied by value.
    {
        lr::Scene        scene;
        lr::SceneObject &object = scene.createSceneObject();
        object.addComponent<lr::TransformComponent>();
        lr::Light &light = object.addComponent<lr::Light>(lr::PointLight{});

        std::unique_ptr<lr::ComponentValues> before = light.undoValues();
        light.set(lr::PointLight{lr::BaseLight{glm::vec3(1.0f), 8.0f}});
        std::unique_ptr<lr::ComponentValues> after = light.undoValues();
        assert(!before->equals(*after));

        lr::CommandManager commands;
        commands.appendCommandWithoutExecuting(std::make_unique<lr::EditComponentValuesCommand>(
            scene, object.id(), lightType, std::move(before), std::move(after)));

        commands.undo();
        assert(intensityOf(object.getComponent<lr::Light>()) == 1.0f);
        commands.redo();
        assert(intensityOf(object.getComponent<lr::Light>()) == 8.0f);
    }

    // The command names its target by type rather than holding the component, so it still reaches
    // the right one after that component has been deleted and added back.
    {
        lr::Scene        scene;
        lr::SceneObject &object = scene.createSceneObject();
        object.addComponent<lr::TransformComponent>();
        lr::Light &light = object.addComponent<lr::Light>(lr::PointLight{});

        std::unique_ptr<lr::ComponentValues> before = light.undoValues();
        light.set(lr::PointLight{lr::BaseLight{glm::vec3(1.0f), 4.0f}});

        lr::CommandManager commands;
        commands.appendCommandWithoutExecuting(std::make_unique<lr::EditComponentValuesCommand>(
            scene, object.id(), lightType, std::move(before), light.undoValues()));

        object.removeComponent(lightType);
        object.addComponent<lr::Light>(lr::PointLight{lr::BaseLight{glm::vec3(1.0f), 9.0f}});
        commands.undo();
        assert(intensityOf(object.getComponent<lr::Light>()) == 1.0f);
    }

    // A target that has gone away since is not written through: undoing is a no-op, not a crash.
    {
        lr::Scene        scene;
        lr::SceneObject &object = scene.createSceneObject();
        object.addComponent<lr::TransformComponent>();
        lr::Light &light = object.addComponent<lr::Light>(lr::PointLight{});

        lr::CommandManager commands;
        commands.appendCommandWithoutExecuting(std::make_unique<lr::EditComponentValuesCommand>(
            scene, object.id(), lightType, light.undoValues(), light.undoValues()));

        const lr::SceneObjectId id = object.id();
        scene.destroySceneObject(id);
        assert(!scene.contains(id));
        commands.undo();
        commands.redo();
    }

    // Deleting a component from the inspector's own menu: undo puts an equivalent one back and
    // announces it, redo takes it off again.
    {
        lr::Scene        scene;
        lr::SceneObject &object = scene.createSceneObject();
        object.addComponent<lr::TransformComponent>();
        lr::Light &light = object.addComponent<lr::Light>(lr::PointLight{lr::BaseLight{glm::vec3(1.0f), 3.0f}});

        CountingAddService addService;
        lr::CommandManager commands;
        commands.executeCommand(std::make_unique<lr::RemoveComponentCommand>(
            scene, object.id(), lightType, light.undoValues(), light.valuesAdder(), addService));
        assert(!object.hasComponent<lr::Light>());

        commands.undo();
        assert(object.hasComponent<lr::Light>());
        assert(intensityOf(object.getComponent<lr::Light>()) == 3.0f);
        assert(addService.additions == 1);

        commands.redo();
        assert(!object.hasComponent<lr::Light>());
    }

    // Adding one: the inspector has already performed the add, so the command is appended rather
    // than executed, and undo is what takes the component off.
    {
        lr::Scene        scene;
        lr::SceneObject &object = scene.createSceneObject();
        object.addComponent<lr::TransformComponent>();
        lr::Camera &camera = object.addComponent<lr::Camera>();
        camera.fovYDegrees = 35.0f;

        const std::type_index cameraType(typeid(lr::Camera));
        CountingAddService    addService;
        lr::CommandManager    commands;
        commands.appendCommandWithoutExecuting(std::make_unique<lr::AddComponentCommand>(
            scene, object.id(), cameraType, camera.undoValues(), camera.valuesAdder(), addService));

        commands.undo();
        assert(!object.hasComponent<lr::Camera>());
        commands.redo();
        assert(object.hasComponent<lr::Camera>());
        // Redoing reproduces the component the user had, not a default one.
        assert(object.getComponent<lr::Camera>().fovYDegrees == 35.0f);
        assert(addService.additions == 1);
    }

    // Renaming from the hierarchy.
    {
        lr::Scene        scene;
        lr::SceneObject &object = scene.createSceneObject();
        object.name             = "Before";

        lr::CommandManager commands;
        commands.executeCommand(
            std::make_unique<lr::RenameSceneObjectCommand>(scene, object.id(), object.name, "After"));
        assert(object.name == "After");
        commands.undo();
        assert(object.name == "Before");
        commands.redo();
        assert(object.name == "After");
    }

    // Snapshots compare by value, which is what keeps an interaction that changed nothing out of
    // the history.
    {
        lr::Scene        scene;
        lr::SceneObject &object    = scene.createSceneObject();
        auto            &transform = object.addComponent<lr::TransformComponent>();

        std::unique_ptr<lr::ComponentValues> first = transform.undoValues();
        assert(first->equals(*transform.undoValues()));
        transform.setPosition(glm::vec3(0.0f, 1.0f, 0.0f));
        assert(!first->equals(*transform.undoValues()));
        transform.setPosition(glm::vec3(0.0f));
        assert(first->equals(*transform.undoValues()));
    }

    return 0;
}
