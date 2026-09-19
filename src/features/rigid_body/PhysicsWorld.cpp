#include "features/rigid_body/PhysicsWorld.hpp"

#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"
#include "features/rigid_body/RigidBodyComponent.hpp"

#include <imgui.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lr
{
namespace
{

constexpr float kStepsPerSecondSmoothingWindow = 0.5f;

void validateConfig(const PhysicsWorld::Config &config)
{
    if (!std::isfinite(config.fixedDeltaTime) || config.fixedDeltaTime <= 0.0f)
    {
        throw std::invalid_argument("Physics fixed timestep must be finite and greater than zero");
    }
    if (!std::isfinite(config.maxFrameTime) || config.maxFrameTime <= 0.0f)
    {
        throw std::invalid_argument("Physics maximum frame time must be finite and greater than zero");
    }
    if (config.maxStepsPerFrame == 0)
    {
        throw std::invalid_argument("Physics maximum steps per frame must be greater than zero");
    }
}

glm::quat calculateWorldRotation(const Scene &scene, const SceneObject &object)
{
    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
    if (object.hasComponent<TransformComponent>())
    {
        rotation = object.getComponent<TransformComponent>().transform().rotation();
    }
    if (object.parent())
    {
        rotation = calculateWorldRotation(scene, scene.getSceneObject(*object.parent())) * rotation;
    }
    return glm::normalize(rotation);
}

} // namespace

PhysicsWorld::PhysicsWorld(Scene &scene) : PhysicsWorld(scene, Config{}) {}

PhysicsWorld::PhysicsWorld(Scene &scene, Config config)
    : m_scene(&scene), m_gravity(config.gravity), m_fixedDeltaTime(config.fixedDeltaTime),
      m_maxFrameTime(config.maxFrameTime), m_maxStepsPerFrame(config.maxStepsPerFrame),
      m_paused(config.startPaused)
{
    validateConfig(config);
    captureInitialState();
}

void PhysicsWorld::captureInitialState()
{
    m_initialStates.clear();
    for (const auto &object : m_scene->sceneObjects())
    {
        if (!object->hasComponent<RigidBodyComponent>() || !object->hasComponent<TransformComponent>())
        {
            continue;
        }

        const Transform &transform = object->getComponent<TransformComponent>().transform();
        const RigidBodyComponent &body = object->getComponent<RigidBodyComponent>();
        m_initialStates.push_back({
            .objectId        = object->id(),
            .localPosition   = transform.position(),
            .localRotation   = transform.rotation(),
            .linearVelocity  = body.linearVelocity(),
            .angularVelocity = body.angularVelocity(),
        });
    }
}

void PhysicsWorld::reset()
{
    setPaused(true);
    m_started       = false;
    m_accumulator   = 0.0f;
    m_lastStepCount = 0;

    for (const InitialBodyState &initial : m_initialStates)
    {
        SceneObject &object = m_scene->getSceneObject(initial.objectId);
        if (!object.hasComponent<RigidBodyComponent>() || !object.hasComponent<TransformComponent>())
        {
            continue;
        }

        TransformComponent &transform = object.getComponent<TransformComponent>();
        RigidBodyComponent &body      = object.getComponent<RigidBodyComponent>();
        transform.setPosition(initial.localPosition);
        transform.setRotation(initial.localRotation);
        body.setLinearVelocity(initial.linearVelocity);
        body.setAngularVelocity(initial.angularVelocity);
        body.clearAccumulators();
    }
}

void PhysicsWorld::setPaused(bool paused)
{
    m_paused = paused;
    if (paused)
    {
        m_accumulator               = 0.0f;
        m_lastStepCount             = 0;
        m_stepsPerSecondWindowElapsed = 0.0f;
        m_stepsPerSecondWindowSteps    = 0;
        m_smoothedStepsPerSecond       = 0.0f;
    }
}

void PhysicsWorld::setFixedDeltaTime(float fixedDeltaTime)
{
    if (!std::isfinite(fixedDeltaTime) || fixedDeltaTime <= 0.0f)
    {
        throw std::invalid_argument("Physics fixed timestep must be finite and greater than zero");
    }
    m_fixedDeltaTime = fixedDeltaTime;
    m_accumulator    = 0.0f;
}

float PhysicsWorld::interpolationAlpha() const
{
    return m_fixedDeltaTime > 0.0f ? m_accumulator / m_fixedDeltaTime : 0.0f;
}

void PhysicsWorld::update(float frameDeltaTime)
{
    m_lastStepCount = 0;
    if (m_paused || !std::isfinite(frameDeltaTime) || frameDeltaTime <= 0.0f)
    {
        return;
    }

    m_accumulator += std::min(frameDeltaTime, m_maxFrameTime);
    while (m_accumulator >= m_fixedDeltaTime && m_lastStepCount < m_maxStepsPerFrame)
    {
        simulateStep(m_fixedDeltaTime);
        m_accumulator -= m_fixedDeltaTime;
        ++m_lastStepCount;
    }

    if (m_lastStepCount == m_maxStepsPerFrame && m_accumulator >= m_fixedDeltaTime)
    {
        m_accumulator = std::fmod(m_accumulator, m_fixedDeltaTime);
    }

    m_stepsPerSecondWindowElapsed += frameDeltaTime;
    m_stepsPerSecondWindowSteps += m_lastStepCount;
    if (m_stepsPerSecondWindowElapsed >= kStepsPerSecondSmoothingWindow)
    {
        m_smoothedStepsPerSecond = static_cast<float>(m_stepsPerSecondWindowSteps) / m_stepsPerSecondWindowElapsed;
        m_stepsPerSecondWindowElapsed = 0.0f;
        m_stepsPerSecondWindowSteps   = 0;
    }
}

void PhysicsWorld::simulateOneStep()
{
    simulateStep(m_fixedDeltaTime);
    m_accumulator   = 0.0f;
    m_lastStepCount = 1;
}

void PhysicsWorld::simulateStep(float deltaTime)
{
    for (const auto &object : m_scene->sceneObjects())
    {
        if (!object->hasComponent<RigidBodyComponent>())
        {
            continue;
        }

        RigidBodyComponent &body = object->getComponent<RigidBodyComponent>();
        if (body.isDynamic() && object->hasComponent<TransformComponent>())
        {
            const glm::vec3 acceleration = m_gravity + body.accumulatedForce() * body.inverseMass();
            // Implicit linear damping models F_drag proportional to -velocity. Unlike a direct
            // multiplier, this is stable for any non-negative drag and timestep, and converges
            // to a terminal velocity of acceleration / drag under constant acceleration.
            const glm::vec3 velocity = (body.linearVelocity() + acceleration * deltaTime) /
                                       (1.0f + body.linearDrag() * deltaTime);
            body.setLinearVelocity(velocity);

            TransformComponent &transform = object->getComponent<TransformComponent>();
            const glm::vec3 worldPosition = glm::vec3(object->worldMatrix()[3]) + velocity * deltaTime;
            if (object->parent())
            {
                const glm::mat4 parentWorld = m_scene->getSceneObject(*object->parent()).worldMatrix();
                transform.setPosition(glm::vec3(glm::inverse(parentWorld) * glm::vec4(worldPosition, 1.0f)));
            }
            else
            {
                transform.setPosition(worldPosition);
            }

            const glm::quat worldRotation = calculateWorldRotation(*m_scene, *object);
            const glm::mat3 rotationMatrix = glm::mat3_cast(worldRotation);
            const glm::mat3 inverseInertiaWorld =
                rotationMatrix * glm::mat3(body.inverseInertiaDiagonal().x, 0.0f, 0.0f,
                                           0.0f, body.inverseInertiaDiagonal().y, 0.0f,
                                           0.0f, 0.0f, body.inverseInertiaDiagonal().z) *
                glm::transpose(rotationMatrix);
            const glm::vec3 angularAcceleration = inverseInertiaWorld * body.accumulatedTorque();
            const glm::vec3 angularVelocity =
                (body.angularVelocity() + angularAcceleration * deltaTime) /
                (1.0f + body.angularDrag() * deltaTime);
            body.setAngularVelocity(angularVelocity);

            if (glm::dot(angularVelocity, angularVelocity) > 0.0f)
            {
                const glm::quat deltaRotation =
                    glm::angleAxis(glm::length(angularVelocity) * deltaTime, glm::normalize(angularVelocity));
                const glm::quat newWorldRotation = glm::normalize(deltaRotation * worldRotation);
                if (object->parent())
                {
                    const glm::quat parentRotation =
                        calculateWorldRotation(*m_scene, m_scene->getSceneObject(*object->parent()));
                    transform.setRotation(glm::normalize(glm::inverse(parentRotation) * newWorldRotation));
                }
                else
                {
                    transform.setRotation(newWorldRotation);
                }
            }
        }

        // Accumulators are impulses-over-one-step inputs. Persistent forces must be reapplied
        // before each fixed update; gravity is maintained separately by PhysicsWorld.
        body.clearAccumulators();
    }
}

void PhysicsWorld::onGUI()
{
    const char *playButtonLabel = !m_started ? "Start" : (m_paused ? "Resume" : "Stop");
    if (ImGui::Button(playButtonLabel))
    {
        if (!m_started)
        {
            // Re-capture here (rather than relying solely on the construction-time snapshot) so
            // Reset restores whatever state the scene was in right before this run started.
            captureInitialState();
            m_started = true;
        }
        setPaused(!m_paused);
    }
    ImGui::SameLine();
    if (ImGui::Button("Step"))
    {
        simulateOneStep();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset"))
    {
        reset();
    }

    ImGui::DragFloat3("Gravity", &m_gravity.x, 0.05f);

    float fixedDeltaTime = m_fixedDeltaTime;
    if (ImGui::DragFloat("Fixed Timestep", &fixedDeltaTime, 0.0001f, 0.0001f, 0.1f, "%.5f s"))
    {
        setFixedDeltaTime(std::max(fixedDeltaTime, 0.0001f));
    }

    int maxSteps = static_cast<int>(m_maxStepsPerFrame);
    if (ImGui::DragInt("Max Steps / Frame", &maxSteps, 1.0f, 1, 64))
    {
        m_maxStepsPerFrame = static_cast<uint32_t>(std::max(maxSteps, 1));
    }

    ImGui::Text("Steps/sec: %.1f", m_smoothedStepsPerSecond);
}

} // namespace lr
