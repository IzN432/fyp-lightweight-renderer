#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"
#include "features/linear_blend_skinning/Skin.hpp"
#include "features/linear_blend_skinning/SkinComponent.hpp"
#include "features/linear_blend_skinning/editor/SkeletonOverlay.hpp"

#include <glm/gtc/quaternion.hpp>

#include <cassert>
#include <cmath>

namespace
{

bool near(const glm::vec3 &a, const glm::vec3 &b)
{
    return glm::length(a - b) < 1e-4f;
}

lr::Joint joint(const lr::SceneObject &object)
{
    return {.sceneObject = object.id()};
}

glm::vec3 arrowEnd(const lr::OverlayInstance &instance)
{
    const glm::quat orientation(glm::radians(instance.eulerDegrees));
    return instance.position + orientation * glm::vec3(0.0f, instance.scale.y, 0.0f);
}

} // namespace

int main()
{
    lr::Scene scene;
    auto &skinObject = scene.createSceneObject();
    auto &root = scene.createSceneObject();
    auto &middle = scene.createSceneObject();
    auto &leaf = scene.createSceneObject();

    root.addComponent<lr::TransformComponent>(glm::vec3(1.0f, 0.0f, 0.0f));
    middle.addComponent<lr::TransformComponent>(glm::vec3(0.0f, 2.0f, 0.0f));
    leaf.addComponent<lr::TransformComponent>(glm::vec3(1.0f, 0.0f, 0.0f));
    scene.setParent(middle.id(), root.id());
    scene.setParent(leaf.id(), middle.id());

    // Palette order intentionally differs from hierarchy order. The two arrows must still follow
    // the scene parent links, then the three joint balls follow palette order.
    skinObject.addComponent<lr::SkinComponent>(lr::Skin(scene, {joint(leaf), joint(root), joint(middle)}));
    const std::vector<lr::OverlayInstance> instances = lr::buildSkeletonOverlay(scene, skinObject.id());

    assert(instances.size() == 5);
    assert(instances[0].primitive == lr::OverlayPrimitive::Arrow); // middle -> leaf (child palette index 0)
    assert(near(instances[0].position, glm::vec3(1.0f, 2.0f, 0.0f)));
    assert(near(arrowEnd(instances[0]), glm::vec3(2.0f, 2.0f, 0.0f)));
    assert(instances[1].primitive == lr::OverlayPrimitive::Arrow); // root -> middle (child palette index 2)
    assert(near(instances[1].position, glm::vec3(1.0f, 0.0f, 0.0f)));
    assert(near(arrowEnd(instances[1]), glm::vec3(1.0f, 2.0f, 0.0f)));

    assert(instances[2].primitive == lr::OverlayPrimitive::Sphere);
    assert(near(instances[2].position, glm::vec3(2.0f, 2.0f, 0.0f))); // leaf, palette index 0
    assert(near(instances[3].position, glm::vec3(1.0f, 0.0f, 0.0f))); // root, palette index 1
    assert(near(instances[4].position, glm::vec3(1.0f, 2.0f, 0.0f))); // middle, palette index 2
    assert(instances[2].pickingId == ~0u);
    assert(instances[2].occludedOpacity > 0.0f);
}
