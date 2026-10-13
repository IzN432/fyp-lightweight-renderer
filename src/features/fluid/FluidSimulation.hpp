#pragma once

#include "features/fluid/FluidSolverGpu.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace lr
{

class OverlayLineBuilder;

// Application-facing GPU fluid controller. It owns simulation setup, source generation, runtime
// controls and the solver, while renderers consume the published particle buffer directly.
class FluidSimulation
{
public:
    struct CuboidSource
    {
        // Number of particles along each axis. The total simulation particle count is the sum of
        // the products of these dimensions across all configured sources.
        glm::uvec3 dimensions{1, 1, 1};
        // World-space centre of the first (minimum-corner) particle in this source lattice.
        glm::vec3 origin{0.0f};
    };

    struct Config
    {
        // Sources can be split into independently positioned cuboids without changing the solver.
        // The default remains the original 50 * 56 * 50 = 140,000-particle block.
        std::vector<CuboidSource> sources{{{50, 56, 50}, {-14.5f, 0.5f, -14.5f}}};
        // Initial GPU allocation only. It grows automatically if an edited layout needs more.
        uint32_t                  maxParticles = 140000;

        // Normalized paper units: h = 2 * spacing = 1. This lets the paper's numerical
        // coefficients be used literally instead of inventing a world-scale conversion.
        float particleSpacing      = 0.370f;
        float smoothingRadiusScale = 2.0f;
        float particleRenderScale  = 0.9f;
        int   solverIterations     = 8;
        // Equations (13) and (17), used literally in normalized h=1 units.
        float artificialPressureK  = 0.02f;
        float vorticityEpsilon     = 0.0f;
        float viscosity            = 0.03f;
        float boundsFriction       = 0.98f;
        // CFL-style safety limit. Half a particle spacing preserves the local neighbour set while
        // still allowing visible motion; zero disables the cap.
        float maxDisplacementScale = 0.5f;

        glm::vec3 boundsMin{-15.0f, 0.0f, -15.0f};
        glm::vec3 boundsMax{15.0f, 35.0f, 15.0f};
        // Presentation-only toggle for the wireframe simulation domain.
        bool      showBounds       = true;
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
    // Appends the presentation-scaled domain outline when Show Bounds is enabled.
    void appendOverlayLines(OverlayLineBuilder &builder, float visualScale) const;

    uint32_t particleCount() const { return m_solver.particleCount(); }
    uint32_t capacity() const { return m_solver.config().maxParticles; }
    uint32_t pendingSteps() const { return m_solver.pendingSteps(); }
    float    particleRadius() const { return m_config.particleRadius(); }
    // Uniform presentation scale whose largest domain dimension is exactly one world unit.
    float unitCubeRenderScale() const;

    const glm::vec3   &particleColor() const { return m_config.particleColor; }
    const glm::vec3   &boundsMin() const { return m_config.boundsMin; }
    const glm::vec3   &boundsMax() const { return m_config.boundsMax; }
    bool               showBounds() const { return m_config.showBounds; }
    const std::string &particleBufferName() const { return m_solver.config().particleBufferResourceName; }

private:
    static FluidSolverGpu::Config solverConfig(const Config &config);
    static std::vector<glm::vec3> makeSource(const Config &config);
    void                          applyRuntimeConfig();

    Config         m_config;
    FluidSolverGpu m_solver;
    bool           m_singleStepRequested = false;
    bool           m_sourcesDirty        = false;
    std::string    m_sourceEditorError;
    std::string    m_error;
};

} // namespace lr
