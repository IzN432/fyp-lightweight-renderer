#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"
#include "core/overlay/PrimitiveOverlayMeshes.hpp"
#include "features/rigid_body/ColliderComponent.hpp"
#include "features/rigid_body/ColliderVisual.hpp"
#include "features/rigid_body/PhysicsWorld.hpp"
#include "features/rigid_body/RigidBodyComponent.hpp"

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
    component.setVisible(true);

    assert(std::holds_alternative<lr::SphereCollider>(component.collider().shape));
    assert(component.collider().material.restitution == 0.5f);
    assert(component.collider().material.staticFriction == 0.5f);
    assert(component.collider().material.dynamicFriction == 0.3f);

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

    lr::RigidBodyComponent body(2.0f);
    assert(body.isDynamic());
    assert(body.mass() == 2.0f);
    assert(body.inverseMass() == 0.5f);
    assert(body.inertiaDiagonal() == glm::vec3(2.0f / 6.0f));
    assert(body.linearDrag() == 0.05f);
    assert(body.angularDrag() == 0.05f);
    body.addForce({1.0f, 2.0f, 3.0f});
    body.addTorque({4.0f, 5.0f, 6.0f});
    assert(body.accumulatedForce() == glm::vec3(1.0f, 2.0f, 3.0f));
    assert(body.accumulatedTorque() == glm::vec3(4.0f, 5.0f, 6.0f));
    body.clearAccumulators();
    assert(body.accumulatedForce() == glm::vec3(0.0f));
    assert(body.accumulatedTorque() == glm::vec3(0.0f));
    body.setType(lr::RigidBodyType::Static);
    assert(body.isStatic());
    assert(body.inverseMass() == 0.0f);
    assert(body.inverseInertiaDiagonal() == glm::vec3(0.0f));
    body.setType(lr::RigidBodyType::Dynamic);
    assert(body.inverseMass() == 0.5f);
    body.setInertiaDiagonal(glm::vec3(2.0f, 4.0f, 8.0f));
    assert(body.inverseInertiaDiagonal() == glm::vec3(0.5f, 0.25f, 0.125f));

    lr::Scene physicsScene;
    auto &fallingObject = physicsScene.createSceneObject();
    auto &fallingTransform = fallingObject.addComponent<lr::TransformComponent>();
    auto &fallingBody = fallingObject.addComponent<lr::RigidBodyComponent>(2.0f);
    fallingBody.setLinearDrag(0.0f);
    fallingBody.setAngularDrag(0.0f);

    auto &staticObject = physicsScene.createSceneObject();
    auto &staticTransform = staticObject.addComponent<lr::TransformComponent>(glm::vec3(0.0f, 5.0f, 0.0f));
    staticObject.addComponent<lr::RigidBodyComponent>(1.0f, lr::RigidBodyType::Static);

    lr::PhysicsWorld physicsWorld(physicsScene, {
        .gravity          = glm::vec3(0.0f, -10.0f, 0.0f),
        .fixedDeltaTime   = 0.1f,
        .maxFrameTime     = 1.0f,
        .maxStepsPerFrame = 10,
        .startPaused      = false,
    });

    physicsWorld.update(0.05f);
    assert(fallingTransform.transform().position() == glm::vec3(0.0f));
    physicsWorld.update(0.05f);
    assert(glm::all(glm::epsilonEqual(fallingBody.linearVelocity(), glm::vec3(0.0f, -1.0f, 0.0f), 0.0001f)));
    assert(glm::all(glm::epsilonEqual(fallingTransform.transform().position(), glm::vec3(0.0f, -0.1f, 0.0f),
                                     0.0001f)));
    assert(staticTransform.transform().position() == glm::vec3(0.0f, 5.0f, 0.0f));

    // Upward force exactly cancels gravity for this two-kilogram body during one fixed step.
    fallingBody.addForce(glm::vec3(0.0f, 20.0f, 0.0f));
    physicsWorld.simulateOneStep();
    assert(glm::all(glm::epsilonEqual(fallingBody.linearVelocity(), glm::vec3(0.0f, -1.0f, 0.0f), 0.0001f)));
    assert(fallingBody.accumulatedForce() == glm::vec3(0.0f));

    fallingBody.setInertiaDiagonal(glm::vec3(2.0f));
    fallingBody.addTorque(glm::vec3(0.0f, 4.0f, 0.0f));
    const glm::quat rotationBeforeTorque = fallingTransform.transform().rotation();
    physicsWorld.simulateOneStep();
    assert(glm::all(glm::epsilonEqual(fallingBody.angularVelocity(), glm::vec3(0.0f, 0.2f, 0.0f), 0.0001f)));
    assert(fallingBody.accumulatedTorque() == glm::vec3(0.0f));
    assert(glm::abs(glm::dot(rotationBeforeTorque, fallingTransform.transform().rotation())) < 0.999999f);

    // Linear and angular drag converge toward rest without becoming unstable.
    fallingBody.setLinearDrag(1.0f);
    fallingBody.setAngularDrag(1.0f);
    fallingBody.setLinearVelocity(glm::vec3(10.0f, 0.0f, 0.0f));
    fallingBody.setAngularVelocity(glm::vec3(0.0f, 10.0f, 0.0f));
    physicsWorld.setGravity(glm::vec3(0.0f));
    physicsWorld.simulateOneStep();
    assert(glm::all(glm::epsilonEqual(fallingBody.linearVelocity(), glm::vec3(10.0f / 1.1f, 0.0f, 0.0f), 0.0001f)));
    assert(glm::all(glm::epsilonEqual(fallingBody.angularVelocity(), glm::vec3(0.0f, 10.0f / 1.1f, 0.0f), 0.0001f)));

    physicsWorld.reset();
    assert(physicsWorld.paused());
    assert(fallingTransform.transform().position() == glm::vec3(0.0f));
    assert(fallingBody.linearVelocity() == glm::vec3(0.0f));
}
