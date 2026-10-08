#include "core/scene/Camera.hpp"
#include "core/scene/ComponentClipboard.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cassert>
#include <typeindex>

namespace
{

// Stands in for any component that has not opted into copy/paste.
struct SilentComponent final : public lr::Component
{
    SilentComponent() : Component("SilentComponent") {}
};

} // namespace

int main()
{
    lr::Scene        scene;
    lr::SceneObject &source = scene.createSceneObject();
    lr::SceneObject &target = scene.createSceneObject();

    auto &sourceTransform = source.addComponent<lr::TransformComponent>(
        glm::vec3(1.0f, 2.0f, 3.0f), glm::quat(glm::radians(glm::vec3(10.0f, 20.0f, 30.0f))),
        glm::vec3(2.0f, 2.0f, 2.0f));
    auto &targetTransform = target.addComponent<lr::TransformComponent>();

    const std::type_index transformType(typeid(lr::TransformComponent));

    lr::ComponentClipboard clipboard;
    assert(!clipboard.holdsValuesFor(transformType));

    clipboard.store(transformType, sourceTransform.copyValues());
    assert(clipboard.holdsValuesFor(transformType));

    targetTransform.clearDirty();
    targetTransform.pasteValues(*clipboard.valuesFor(transformType));
    assert(targetTransform.isDirty());
    assert(glm::all(glm::epsilonEqual(targetTransform.transform().position(),
                                      sourceTransform.transform().position(), 0.0001f)));
    assert(glm::all(glm::epsilonEqual(targetTransform.transform().scale(),
                                      sourceTransform.transform().scale(), 0.0001f)));
    // Pasting carries the euler angles the inspector shows, not just the quaternion they came from.
    assert(glm::all(glm::epsilonEqual(targetTransform.transform().eulerDegrees(),
                                      sourceTransform.transform().eulerDegrees(), 0.0001f)));

    // The snapshot is detached: editing the source afterwards does not change what gets pasted.
    sourceTransform.setPosition(glm::vec3(-5.0f));
    lr::SceneObject &second          = scene.createSceneObject();
    auto            &secondTransform = second.addComponent<lr::TransformComponent>();
    secondTransform.pasteValues(*clipboard.valuesFor(transformType));
    assert(glm::all(glm::epsilonEqual(secondTransform.transform().position(), glm::vec3(1.0f, 2.0f, 3.0f), 0.0001f)));

    // A paste is only ever offered back to the component type that produced the snapshot.
    assert(clipboard.valuesFor(std::type_index(typeid(SilentComponent))) == nullptr);

    // Components that have not opted in stay out of the menu entirely.
    SilentComponent silent;
    assert(silent.copyValues() == nullptr);

    // Storing again replaces the previous snapshot rather than accumulating one per type.
    clipboard.store(std::type_index(typeid(SilentComponent)), nullptr);
    assert(!clipboard.holdsValuesFor(transformType));
    assert(!clipboard.holdsValuesFor(std::type_index(typeid(SilentComponent))));

    // A light's snapshot carries its type as well as its parameters, so pasting a spot light over a
    // point light turns the target into a spot light.
    const std::type_index lightType(typeid(lr::Light));
    lr::SceneObject      &spotObject = scene.createSceneObject();
    lr::SpotLight         spot;
    spot.color                 = glm::vec3(0.2f, 0.4f, 0.9f);
    spot.intensity             = 12.0f;
    spot.outerConeAngleDegrees = 42.0f;
    spot.innerConeAngleDegrees = 11.0f;
    auto &spotLight            = spotObject.addComponent<lr::Light>(spot);

    lr::SceneObject &pointObject = scene.createSceneObject();
    auto            &pointLight  = pointObject.addComponent<lr::Light>(lr::PointLight{});
    assert(std::holds_alternative<lr::PointLight>(pointLight.light));

    clipboard.store(lightType, spotLight.copyValues());
    pointLight.pasteValues(*clipboard.valuesFor(lightType));
    const auto *pastedSpot = std::get_if<lr::SpotLight>(&pointLight.light);
    assert(pastedSpot != nullptr);
    assert(glm::all(glm::epsilonEqual(pastedSpot->color, spot.color, 0.0001f)));
    assert(pastedSpot->intensity == spot.intensity);
    assert(pastedSpot->outerConeAngleDegrees == spot.outerConeAngleDegrees);
    assert(pastedSpot->innerConeAngleDegrees == spot.innerConeAngleDegrees);

    // A light snapshot is never offered back to a different component type, however similar.
    assert(clipboard.valuesFor(transformType) == nullptr);

    // A camera copies its projection; its placement belongs to the object's transform instead.
    const std::type_index cameraType(typeid(lr::Camera));
    lr::SceneObject      &cameraObject = scene.createSceneObject();
    auto                 &sourceCamera = cameraObject.addComponent<lr::Camera>();
    sourceCamera.projectionType        = lr::ProjectionType::Orthographic;
    sourceCamera.orthoHeight           = 24.0f;
    sourceCamera.nearPlane             = 0.5f;
    sourceCamera.farPlane              = 250.0f;
    sourceCamera.fovYDegrees           = 35.0f;

    lr::SceneObject &otherCameraObject = scene.createSceneObject();
    auto            &targetCamera      = otherCameraObject.addComponent<lr::Camera>();
    clipboard.store(cameraType, sourceCamera.copyValues());
    targetCamera.pasteValues(*clipboard.valuesFor(cameraType));
    assert(targetCamera.projectionType == lr::ProjectionType::Orthographic);
    assert(targetCamera.orthoHeight == 24.0f);
    assert(targetCamera.nearPlane == 0.5f);
    assert(targetCamera.farPlane == 250.0f);
    assert(targetCamera.fovYDegrees == 35.0f);
}
