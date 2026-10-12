#pragma once

#include "features/fluid/FluidSolverGpu.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace lr
{

// Application-facing GPU fluid controller. It owns simulation setup, source generation, runtime
// controls and the solver, while renderers consume the published particle buffer directly.
class FluidSimulation
{
public:
    struct Config
    {
        // 50 * 56 * 50 = exactly 140,000 particles.
        glm::uvec3 sourceDimensions{50, 56, 50};
        uint32_t   maxParticles = 140000;

        // Normalized paper units: h = 2 * spacing = 1. This lets the paper's numerical
        // coefficients be used literally instead of inventing a world-scale conversion.
        float particleSpacing      = 0.5f;
        float smoothingRadiusScale = 2.0f;
        float particleRenderScale  = 0.9f;
        int   solverIterations     = 8;
        // Equations (13) and (17), used literally in normalized h=1 units.
        float artificialPressureK  = 0.1f;
        float vorticityEpsilon     = 0.0f;
        float viscosity            = 0.01f;
        float boundsFriction       = 0.98f;

        glm::vec3 boundsMin{-15.0f, 0.0f, -15.0f};
        glm::vec3 boundsMax{15.0f, 35.0f, 15.0f};
        float     fixedDeltaTime   = 0.016f;
        int       maxStepsPerFrame = 2;
        bool      startPaused      = false;
        glm::vec3 particleColor{0.25f, 0.5f, 0.95f};

        float smoothingRadius() const;
        float particleRadius() const;
    };

    FluidSimulation(Config config, ResourceRegistry &resources);

    void build(FrameGraph &frameGraph);
    void update(float frameDeltaTime);
    void reset();
    void onGUI();

    uint32_t particleCount() const { return m_solver.particleCount(); }
    uint32_t capacity() const { return m_config.maxParticles; }
    uint32_t pendingSteps() const { return m_solver.pendingSteps(); }
    float    particleRadius() const { return m_config.particleRadius(); }

    const glm::vec3   &particleColor() const { return m_config.particleColor; }
    const glm::vec3   &boundsMin() const { return m_config.boundsMin; }
    const glm::vec3   &boundsMax() const { return m_config.boundsMax; }
    const std::string &particleBufferName() const { return m_solver.config().particleBufferResourceName; }

private:
    static FluidSolverGpu::Config solverConfig(const Config &config);
    static std::vector<glm::vec3> makeSource(const Config &config);
    void                          applyRuntimeConfig();

    Config         m_config;
    FluidSolverGpu m_solver;
    bool           m_singleStepRequested = false;
    std::string    m_error;
};

} // namespace lr
