#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"
#include "features/linear_blend_skinning/Joint.hpp"
#include "features/linear_blend_skinning/Skin.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>

#include <cassert>
#include <stdexcept>
#include <type_traits>

int main()
{
    static_assert(std::is_same_v<decltype(lr::Joint::sceneObject), lr::SceneObjectId>);

    lr::Scene        scene;
    lr::SceneObject &root = scene.createSceneObject();
    root.addComponent<lr::TransformComponent>().setPosition(glm::vec3(1.0f, 2.0f, 3.0f));

    lr::SceneObject &jointObject = scene.createSceneObject();
    jointObject.addComponent<lr::TransformComponent>().setPosition(glm::vec3(0.0f, 4.0f, 0.0f));
    scene.setParent(jointObject.id(), root.id());

    const lr::Joint joint{.sceneObject = jointObject.id(), .inverseBindMatrix = glm::mat4(1.0f)};
    lr::Skin        skin(scene, {joint});
    assert(skin.joints().size() == 1);
    assert(skin.joints()[0].sceneObject == jointObject.id());

    skin.evaluate(glm::mat4(1.0f));
    assert(glm::all(glm::epsilonEqual(glm::vec3(skin.jointMatrices()[0][3]),
                                     glm::vec3(1.0f, 6.0f, 3.0f), 0.0001f)));

    // Posing happens on the scene object's local transform; Skin only reads
    // the resulting world matrix when rebuilding the palette.
    jointObject.getComponent<lr::TransformComponent>().setPosition(glm::vec3(5.0f, 0.0f, 0.0f));
    skin.evaluate(glm::mat4(1.0f));
    assert(glm::all(glm::epsilonEqual(glm::vec3(skin.jointMatrices()[0][3]),
                                     glm::vec3(6.0f, 2.0f, 3.0f), 0.0001f)));

    // Joint matrices are expressed in mesh-local space.
    const glm::mat4 meshWorld = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    skin.evaluate(meshWorld);
    assert(glm::all(glm::epsilonEqual(glm::vec3(skin.jointMatrices()[0][3]),
                                     glm::vec3(5.0f, 2.0f, 3.0f), 0.0001f)));

    bool invalidJointRejected = false;
    try
    {
        lr::Skin invalid(scene, {{.sceneObject = 1000}});
    }
    catch (const std::invalid_argument &)
    {
        invalidJointRejected = true;
    }
    assert(invalidJointRejected);
}
