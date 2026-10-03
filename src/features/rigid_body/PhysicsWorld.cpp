#include "features/rigid_body/PhysicsWorld.hpp"

#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"
#include "features/rigid_body/ColliderComponent.hpp"
#include "features/rigid_body/RigidBodyComponent.hpp"

#include <reactphysics3d/reactphysics3d.h>

#include <imgui.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <variant>

namespace lr
{
namespace
{
namespace rp3d = reactphysics3d;
constexpr float kStepsPerSecondSmoothingWindow = 0.5f;
constexpr float kPlaneThickness = 0.01f;

rp3d::Vector3 toRp3d(const glm::vec3 &v) { return {v.x, v.y, v.z}; }
glm::vec3 toGlm(const rp3d::Vector3 &v) { return {v.x, v.y, v.z}; }
rp3d::Quaternion toRp3d(const glm::quat &q) { return {q.x, q.y, q.z, q.w}; }
glm::quat toGlm(const rp3d::Quaternion &q) { return {q.w, q.x, q.y, q.z}; }

void validateConfig(const PhysicsWorld::Config &config)
{
    if (!std::isfinite(config.fixedDeltaTime) || config.fixedDeltaTime <= 0.0f)
        throw std::invalid_argument("Physics fixed timestep must be finite and greater than zero");
    if (!std::isfinite(config.maxFrameTime) || config.maxFrameTime <= 0.0f)
        throw std::invalid_argument("Physics maximum frame time must be finite and greater than zero");
    if (config.maxStepsPerFrame == 0)
        throw std::invalid_argument("Physics maximum steps per frame must be greater than zero");
}

glm::quat worldRotation(const Scene &scene, const SceneObject &object)
{
    glm::quat result = object.hasComponent<TransformComponent>()
                           ? object.getComponent<TransformComponent>().transform().rotation()
                           : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    if (object.parent()) result = worldRotation(scene, scene.getSceneObject(*object.parent())) * result;
    return glm::normalize(result);
}

glm::vec3 worldScale(const SceneObject &object)
{
    const glm::mat4 matrix = object.worldMatrix();
    return {glm::length(glm::vec3(matrix[0])), glm::length(glm::vec3(matrix[1])),
            glm::length(glm::vec3(matrix[2]))};
}

rp3d::Transform bodyTransform(const Scene &scene, const SceneObject &object)
{
    return {toRp3d(glm::vec3(object.worldMatrix()[3])), toRp3d(worldRotation(scene, object))};
}
} // namespace

struct PhysicsWorld::Impl
{
    struct BodyRecord
    {
        struct ShapeRecord
        {
            rp3d::CollisionShape *shape{};
            bool sphere{};
        };

        SceneObjectId objectId;
        rp3d::RigidBody *body{};
        std::vector<ShapeRecord> shapes;
        glm::vec3 lastLinearVelocity{0.0f};
        glm::vec3 lastAngularVelocity{0.0f};
    };

    rp3d::PhysicsCommon common;
    rp3d::PhysicsWorld *world{};
    std::vector<BodyRecord> bodies;

    explicit Impl(const glm::vec3 &gravity)
    {
        rp3d::PhysicsWorld::WorldSettings settings;
        settings.gravity = toRp3d(gravity);
        world = common.createPhysicsWorld(settings);
    }

    ~Impl()
    {
        if (world) common.destroyPhysicsWorld(world);
        for (const BodyRecord &record : bodies)
        {
            for (const BodyRecord::ShapeRecord &shape : record.shapes)
            {
                if (shape.sphere)
                    common.destroySphereShape(static_cast<rp3d::SphereShape *>(shape.shape));
                else
                    common.destroyBoxShape(static_cast<rp3d::BoxShape *>(shape.shape));
            }
        }
    }
};

PhysicsWorld::PhysicsWorld(Scene &scene) : PhysicsWorld(scene, Config{}) {}

PhysicsWorld::PhysicsWorld(Scene &scene, Config config)
    : m_scene(&scene), m_gravity(config.gravity), m_fixedDeltaTime(config.fixedDeltaTime),
      m_maxFrameTime(config.maxFrameTime), m_maxStepsPerFrame(config.maxStepsPerFrame),
      m_paused(config.startPaused)
{
    validateConfig(config);
    captureInitialState();
    rebuildBackend();
}

PhysicsWorld::~PhysicsWorld() = default;

void PhysicsWorld::onSceneChanged()
{
    captureInitialState();
    rebuildBackend();
}

void PhysicsWorld::rebuildBackend()
{
    m_impl = std::make_unique<Impl>(m_gravity);
    for (const auto &objectPtr : m_scene->sceneObjects())
    {
        if (!m_scene->contains(objectPtr->id())) continue;
        SceneObject &object = *objectPtr;
        if (!object.hasComponent<RigidBodyComponent>() || !object.hasComponent<ColliderComponent>() ||
            !object.hasComponent<TransformComponent>()) continue;

        const glm::vec3 scale = worldScale(object);
        rp3d::RigidBody *backendBody = m_impl->world->createRigidBody(bodyTransform(*m_scene, object));
        Impl::BodyRecord record{.objectId = object.id(), .body = backendBody};
        for (const Collider &collider : object.getComponent<ColliderComponent>().colliders())
        {
            rp3d::CollisionShape *shape = nullptr;
            bool isSphere = false;
            glm::vec3 localPosition = collider.localPosition * scale;
            if (const auto *sphere = std::get_if<SphereCollider>(&collider.shape))
            {
                isSphere = true;
                shape = m_impl->common.createSphereShape(
                    sphere->radius * std::max({std::abs(scale.x), std::abs(scale.y), std::abs(scale.z)}));
            }
            else if (const auto *box = std::get_if<BoxCollider>(&collider.shape))
                shape = m_impl->common.createBoxShape(toRp3d(glm::abs(box->halfExtents * scale)));
            else
            {
                const auto &plane = std::get<PlaneCollider>(collider.shape);
                shape = m_impl->common.createBoxShape({std::abs(plane.halfExtents.x * scale.x),
                                                       kPlaneThickness,
                                                       std::abs(plane.halfExtents.y * scale.z)});
                localPosition += collider.localRotation * glm::vec3(0.0f, plane.offset * scale.y, 0.0f);
            }
            rp3d::Collider *backendCollider = backendBody->addCollider(
                shape, rp3d::Transform(toRp3d(localPosition), toRp3d(collider.localRotation)));
            backendCollider->getMaterial().setBounciness(
                std::clamp(collider.material.restitution, 0.0f, 1.0f));
            backendCollider->getMaterial().setFrictionCoefficient(
                std::max(collider.material.friction, 0.0f));
            record.shapes.push_back({shape, isSphere});
        }

        RigidBodyComponent &body = object.getComponent<RigidBodyComponent>();
        backendBody->setType(body.isStatic() ? rp3d::BodyType::STATIC : rp3d::BodyType::DYNAMIC);
        backendBody->setMass(body.mass());
        backendBody->setLocalInertiaTensor(toRp3d(body.inertiaDiagonal()));
        backendBody->setLinearDamping(body.linearDrag());
        backendBody->setAngularDamping(body.angularDrag());
        backendBody->setLinearVelocity(toRp3d(body.linearVelocity()));
        backendBody->setAngularVelocity(toRp3d(body.angularVelocity()));
        record.lastLinearVelocity = body.linearVelocity();
        record.lastAngularVelocity = body.angularVelocity();
        m_impl->bodies.push_back(std::move(record));
    }
}

void PhysicsWorld::captureInitialState()
{
    m_initialStates.clear();
    for (const auto &object : m_scene->sceneObjects())
    {
        if (!m_scene->contains(object->id())) continue;
        if (!object->hasComponent<RigidBodyComponent>() || !object->hasComponent<TransformComponent>()) continue;
        const Transform &transform = object->getComponent<TransformComponent>().transform();
        const RigidBodyComponent &body = object->getComponent<RigidBodyComponent>();
        m_initialStates.push_back({object->id(), transform.position(), transform.rotation(),
                                   body.linearVelocity(), body.angularVelocity()});
    }
}

void PhysicsWorld::reset()
{
    setPaused(true);
    m_started = false;
    m_accumulator = 0.0f;
    m_lastStepCount = 0;
    for (const InitialBodyState &initial : m_initialStates)
    {
        SceneObject &object = m_scene->getSceneObject(initial.objectId);
        if (!object.hasComponent<RigidBodyComponent>() || !object.hasComponent<TransformComponent>()) continue;
        auto &transform = object.getComponent<TransformComponent>();
        auto &body = object.getComponent<RigidBodyComponent>();
        transform.setPosition(initial.localPosition);
        transform.setRotation(initial.localRotation);
        body.setLinearVelocity(initial.linearVelocity);
        body.setAngularVelocity(initial.angularVelocity);
        body.clearAccumulators();
    }
    rebuildBackend();
}

void PhysicsWorld::setPaused(bool paused)
{
    m_paused = paused;
    if (paused)
    {
        m_accumulator = 0.0f;
        m_lastStepCount = 0;
        m_stepsPerSecondWindowElapsed = 0.0f;
        m_stepsPerSecondWindowSteps = 0;
        m_smoothedStepsPerSecond = 0.0f;
    }
}

void PhysicsWorld::setFixedDeltaTime(float fixedDeltaTime)
{
    if (!std::isfinite(fixedDeltaTime) || fixedDeltaTime <= 0.0f)
        throw std::invalid_argument("Physics fixed timestep must be finite and greater than zero");
    m_fixedDeltaTime = fixedDeltaTime;
    m_accumulator = 0.0f;
}

float PhysicsWorld::interpolationAlpha() const
{
    return m_fixedDeltaTime > 0.0f ? m_accumulator / m_fixedDeltaTime : 0.0f;
}

void PhysicsWorld::update(float frameDeltaTime)
{
    m_lastStepCount = 0;
    if (m_paused || !std::isfinite(frameDeltaTime) || frameDeltaTime <= 0.0f) return;
    m_accumulator += std::min(frameDeltaTime, m_maxFrameTime);
    while (m_accumulator >= m_fixedDeltaTime && m_lastStepCount < m_maxStepsPerFrame)
    {
        simulateStep(m_fixedDeltaTime);
        m_accumulator -= m_fixedDeltaTime;
        ++m_lastStepCount;
    }
    if (m_lastStepCount == m_maxStepsPerFrame && m_accumulator >= m_fixedDeltaTime)
        m_accumulator = std::fmod(m_accumulator, m_fixedDeltaTime);
    m_stepsPerSecondWindowElapsed += frameDeltaTime;
    m_stepsPerSecondWindowSteps += m_lastStepCount;
    if (m_stepsPerSecondWindowElapsed >= kStepsPerSecondSmoothingWindow)
    {
        m_smoothedStepsPerSecond = static_cast<float>(m_stepsPerSecondWindowSteps) /
                                   m_stepsPerSecondWindowElapsed;
        m_stepsPerSecondWindowElapsed = 0.0f;
        m_stepsPerSecondWindowSteps = 0;
    }
}

void PhysicsWorld::simulateOneStep()
{
    simulateStep(m_fixedDeltaTime);
    m_accumulator = 0.0f;
    m_lastStepCount = 1;
}

void PhysicsWorld::simulateStep(float deltaTime)
{
    m_impl->world->setGravity(toRp3d(m_gravity));
    for (Impl::BodyRecord &record : m_impl->bodies)
    {
        SceneObject &object = m_scene->getSceneObject(record.objectId);
        RigidBodyComponent &body = object.getComponent<RigidBodyComponent>();
        record.body->setType(body.isStatic() ? rp3d::BodyType::STATIC : rp3d::BodyType::DYNAMIC);
        record.body->setMass(body.mass());
        record.body->setLocalInertiaTensor(toRp3d(body.inertiaDiagonal()));
        record.body->setLinearDamping(body.linearDrag());
        record.body->setAngularDamping(body.angularDrag());
        // Only push velocity when application code changed it. Writing the solved velocity back
        // into ReactPhysics3D every step would continually wake otherwise sleeping bodies.
        if (body.linearVelocity() != record.lastLinearVelocity)
            record.body->setLinearVelocity(toRp3d(body.linearVelocity()));
        if (body.angularVelocity() != record.lastAngularVelocity)
            record.body->setAngularVelocity(toRp3d(body.angularVelocity()));
        if (body.isDynamic())
        {
            record.body->applyWorldForceAtCenterOfMass(toRp3d(body.accumulatedForce()));
            record.body->applyWorldTorque(toRp3d(body.accumulatedTorque()));
        }
        body.clearAccumulators();
    }
    m_impl->world->update(deltaTime);

    for (Impl::BodyRecord &record : m_impl->bodies)
    {
        SceneObject &object = m_scene->getSceneObject(record.objectId);
        RigidBodyComponent &body = object.getComponent<RigidBodyComponent>();
        record.lastLinearVelocity = toGlm(record.body->getLinearVelocity());
        record.lastAngularVelocity = toGlm(record.body->getAngularVelocity());
        body.setLinearVelocity(record.lastLinearVelocity);
        body.setAngularVelocity(record.lastAngularVelocity);
        if (body.isStatic()) continue;

        const rp3d::Transform &solved = record.body->getTransform();
        glm::vec3 position = toGlm(solved.getPosition());
        glm::quat rotation = glm::normalize(toGlm(solved.getOrientation()));
        TransformComponent &transform = object.getComponent<TransformComponent>();
        if (object.parent())
        {
            const SceneObject &parent = m_scene->getSceneObject(*object.parent());
            transform.setPosition(glm::vec3(glm::inverse(parent.worldMatrix()) * glm::vec4(position, 1.0f)));
            transform.setRotation(glm::normalize(glm::inverse(worldRotation(*m_scene, parent)) * rotation));
        }
        else
        {
            transform.setPosition(position);
            transform.setRotation(rotation);
        }
    }
}

void PhysicsWorld::onGUI()
{
    const char *label = !m_started ? "Start" : (m_paused ? "Resume" : "Stop");
    if (ImGui::Button(label))
    {
        if (!m_started)
        {
            captureInitialState();
            rebuildBackend();
            m_started = true;
        }
        setPaused(!m_paused);
    }
    ImGui::SameLine();
    if (ImGui::Button("Step")) simulateOneStep();
    ImGui::SameLine();
    if (ImGui::Button("Reset")) reset();
    ImGui::DragFloat3("Gravity", &m_gravity.x, 0.05f);
    float fixedDeltaTime = m_fixedDeltaTime;
    if (ImGui::DragFloat("Fixed Timestep", &fixedDeltaTime, 0.0001f, 0.0001f, 0.1f, "%.5f s"))
        setFixedDeltaTime(std::max(fixedDeltaTime, 0.0001f));
    int maxSteps = static_cast<int>(m_maxStepsPerFrame);
    if (ImGui::DragInt("Max Steps / Frame", &maxSteps, 1.0f, 1, 64))
        m_maxStepsPerFrame = static_cast<uint32_t>(std::max(maxSteps, 1));
    ImGui::Text("Steps/sec: %.1f", m_smoothedStepsPerSecond);
}
} // namespace lr
