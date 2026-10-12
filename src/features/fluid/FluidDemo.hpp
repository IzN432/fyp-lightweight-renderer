#pragma once

#include "core/passes/overlaygeometry/OverlayInstance.hpp"
#include "features/fluid/FluidSolverCpu.hpp"

#include <glm/glm.hpp>

#include <vector>

namespace lr
{

// Drives the CPU fluid solver and turns its particles into overlay sphere instances so the sim can
// be watched in the viewer. This is a scaffold for bringing the solver up, not the shipping render
// path: the overlay pass issues one draw per instance, so particle counts have to stay small. The
// instanced particle pass replaces it once the solver is trusted.
class FluidDemo
{
public:
    struct Config
    {
        // Spacing is the source of truth: smoothing radius, rest density and the drawn particle
        // size are all derived from it so the sim cannot be set up inconsistently.
        float particleSpacing = 0.05f;
        // Multiple of spacing used as the kernel support radius. Around 2 gives each particle the
        // ~30-40 neighbours SPH needs to estimate density well.
        float smoothingRadiusScale = 2.0f;
        // Drawn sphere diameter as a multiple of spacing. Below 1 leaves visible gaps; above 1
        // overlaps into a solid mass.
        float particleRenderScale = 0.9f;
        int   solverIterations    = 8;

        // Forwarded to the solver; see FluidSolverCpu::Config for what each one is for.
        float viscosity     = 0.05f;
        float boundsFriction = 0.98f;
        float boundsDensity  = 1.0f;

        // Simulation domain. The fluid is dropped into one corner of it.
        glm::vec3 boundsMin{-0.5f, 0.0f, -0.5f};
        glm::vec3 boundsMax{0.5f, 1.0f, 0.5f};
        // Fraction of the domain the initial block of fluid occupies, per axis.
        glm::vec3 sourceFraction{0.4f, 0.6f, 0.4f};

        // Fixed simulation timestep, independent of frame rate.
        float fixedDeltaTime = 1.0f / 60.0f;
        // Ceiling on steps per frame so a slow frame cannot spiral into an ever-growing backlog.
        int  maxStepsPerFrame = 4;
        bool startPaused      = true;

        glm::vec3 particleColor{0.25f, 0.5f, 0.95f};
    };

    explicit FluidDemo(Config config);

    // Rebuilds the particle block and clears velocities.
    void reset();

    // Advances the solver by whole fixed steps covering frameDeltaTime.
    void update(float frameDeltaTime);
    void stepOnce();

    // Overlay instances for this frame's particle positions, plus the domain's wireframe box.
    std::vector<OverlayInstance> buildOverlayInstances() const;

    bool paused() const { return m_paused; }
    void setPaused(bool paused) { m_paused = paused; }

    std::size_t particleCount() const { return m_solver.particleCount(); }
    // Mean density over rest density. 1.0 means converged; well above means overcompressed.
    float meanDensityRatio() const;

    const FluidSolverCpu &solver() const { return m_solver; }
    const Config         &config() const { return m_config; }

    // Draws controls only; the caller owns the containing ImGui window.
    void onGUI();

private:
    void rebuildSolverConfig();

    Config         m_config;
    FluidSolverCpu m_solver;
    bool           m_paused;
    float          m_stepAccumulator = 0.0f;
};

} // namespace lr
