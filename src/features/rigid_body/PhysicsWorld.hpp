#pragma once

#include "core/scene/SceneObjectId.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace lr
{

class Scene;

class PhysicsWorld
{
public:
    struct Config
    {
        glm::vec3 gravity{0.0f, -9.81f, 0.0f};
        float     fixedDeltaTime  = 1.0f / 60.0f;
        float     maxFrameTime    = 0.25f;
        uint32_t  maxStepsPerFrame = 8;
        bool      startPaused      = true;
    };

    explicit PhysicsWorld(Scene &scene);
    PhysicsWorld(Scene &scene, Config config);
    ~PhysicsWorld();

    PhysicsWorld(const PhysicsWorld &) = delete;
    PhysicsWorld &operator=(const PhysicsWorld &) = delete;

    // Called once per rendered frame. Executes zero or more fixed simulation steps.
    void update(float frameDeltaTime);
    void simulateOneStep();

    void captureInitialState();
    void reset();

    bool  paused() const { return m_paused; }
    void  setPaused(bool paused);
    bool  started() const { return m_started; }
    float fixedDeltaTime() const { return m_fixedDeltaTime; }
    void  setFixedDeltaTime(float fixedDeltaTime);
    const glm::vec3 &gravity() const { return m_gravity; }
    void             setGravity(const glm::vec3 &gravity) { m_gravity = gravity; }

    float interpolationAlpha() const;
    uint32_t lastStepCount() const { return m_lastStepCount; }
    float stepsPerSecond() const { return m_smoothedStepsPerSecond; }

    // Draws controls only; the caller owns the containing ImGui window.
    void onGUI();

private:
    struct InitialBodyState
    {
        SceneObjectId objectId;
        glm::vec3     localPosition;
        glm::quat     localRotation;
        glm::vec3     linearVelocity;
        glm::vec3     angularVelocity;
    };

    void simulateStep(float deltaTime);
    void rebuildBackend();

    struct Impl;

    Scene                    *m_scene;
    glm::vec3                 m_gravity;
    float                     m_fixedDeltaTime;
    float                     m_maxFrameTime;
    uint32_t                  m_maxStepsPerFrame;
    bool                      m_paused;
    bool                      m_started = false;
    float                     m_accumulator  = 0.0f;
    uint32_t                  m_lastStepCount = 0;

    // Rolling window used to smooth the steps/second readout shown in onGUI, since per-frame
    // step counts flicker between 0 and 1 whenever the frame rate is close to the fixed rate.
    float                     m_stepsPerSecondWindowElapsed = 0.0f;
    uint32_t                  m_stepsPerSecondWindowSteps    = 0;
    float                     m_smoothedStepsPerSecond       = 0.0f;

    std::vector<InitialBodyState> m_initialStates;
    std::unique_ptr<Impl>         m_impl;
};

} // namespace lr
