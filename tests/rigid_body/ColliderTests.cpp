#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"
#include "core/overlay/PrimitiveOverlayMeshes.hpp"
#include "features/rigid_body/ColliderComponent.hpp"
#include "features/rigid_body/ColliderVisual.hpp"

#include <glm/gtc/epsilon.hpp>

#include <cassert>

int main()
{
    assert(!lr::primitives::cube.surface.faces.empty());
    assert(lr::primitives::cube.outline.edges.size() == 12);
    assert(!lr::primitives::sphere.surface.faces.empty());
    assert(lr::primitives::sphere.outline.edges.size() == 128);

    lr::Scene scene;
    auto     &object = scene.createSceneObject();
    object.addComponent<lr::TransformComponent>(glm::vec3(2.0f, 3.0f, 4.0f));

    lr::Collider collider;
    collider.shape         = lr::SphereCollider{2.0f};
    collider.localPosition = glm::vec3(1.0f, 0.0f, 0.0f);
    auto &component = object.addComponent<lr::ColliderComponent>(collider);

    assert(std::holds_alternative<lr::SphereCollider>(component.collider().shape));
    assert(component.collider().restitution == 0.5f);
    assert(component.collider().friction == 0.5f);

    const auto sphereLines = lr::buildColliderOverlayLines(scene);
    assert(sphereLines.size() == 128); // Two 64-segment great circles.
    assert(glm::all(glm::epsilonEqual(sphereLines.front().start,
                                     glm::vec3(5.0f, 3.0f, 4.0f), 0.0001f)));

    component.collider().shape = lr::BoxCollider{glm::vec3(1.0f, 2.0f, 3.0f)};
    const auto boxLines = lr::buildColliderOverlayLines(scene);
    assert(boxLines.size() == 12);

    component.collider().shape = lr::PlaneCollider{
        .offset      = 0.0f,
        .halfExtents = glm::vec2(2.0f, 3.0f),
    };
    const auto planeLines = lr::buildColliderOverlayLines(scene);
    assert(planeLines.size() == 4); // Four finite edges.
    assert(glm::all(glm::epsilonEqual(planeLines.front().start,
                                     glm::vec3(1.0f, 3.0f, 1.0f), 0.0001f)));
    assert(glm::all(glm::epsilonEqual(planeLines.front().end,
                                     glm::vec3(5.0f, 3.0f, 1.0f), 0.0001f)));
}
