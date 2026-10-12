#include "features/fluid/FluidDemo.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>

namespace lr
{

FluidDemo::FluidDemo(Config config) : m_config(config), m_paused(config.startPaused)
{
    reset();
}

void FluidDemo::rebuildSolverConfig()
{
    const float spacing = std::max(m_config.particleSpacing, 1e-4f);
    const float radius  = spacing * std::max(m_config.smoothingRadiusScale, 1.0f);

    FluidSolverCpu::Config solverConfig;
    solverConfig.smoothingRadius = radius;
    solverConfig.particleMass    = 1.0f;
    // Derived from the lattice the particles actually start on rather than chosen independently,
    // which is what keeps the block from inflating or collapsing on the first step.
    solverConfig.restDensity      = FluidSolverCpu::restDensityForSpacing(spacing, radius, 1.0f);
    solverConfig.relaxation       = solverConfig.restDensity * 0.01f;
    solverConfig.solverIterations = std::max(m_config.solverIterations, 0);
    solverConfig.viscosity        = m_config.viscosity;
    solverConfig.boundsFriction   = m_config.boundsFriction;
    solverConfig.boundsDensity    = m_config.boundsDensity;
    solverConfig.boundsMin        = m_config.boundsMin;
    solverConfig.boundsMax        = m_config.boundsMax;
    m_solver.setConfig(solverConfig);
}

void FluidDemo::reset()
{
    rebuildSolverConfig();

    const float     spacing = std::max(m_config.particleSpacing, 1e-4f);
    const glm::vec3 extent  = m_config.boundsMax - m_config.boundsMin;
    // Inset by one spacing so no particle starts exactly on a wall, where the boundary clamp would
    // fight the density solve on the very first step.
    const glm::vec3 min = m_config.boundsMin + glm::vec3(spacing);
    const glm::vec3 max = min + extent * m_config.sourceFraction;

    m_solver.setParticles(FluidSolverCpu::makeParticleBlock(min, max, spacing));
    m_stepAccumulator = 0.0f;
    m_paused          = m_config.startPaused;
}

void FluidDemo::stepOnce()
{
    m_solver.step(std::max(m_config.fixedDeltaTime, 1e-5f));
}

void FluidDemo::update(float frameDeltaTime)
{
    if (m_paused || frameDeltaTime <= 0.0f)
    {
        return;
    }

    const float step = std::max(m_config.fixedDeltaTime, 1e-5f);
    m_stepAccumulator += frameDeltaTime;

    const int maxSteps = std::max(m_config.maxStepsPerFrame, 1);
    int       steps    = 0;
    while (m_stepAccumulator >= step && steps < maxSteps)
    {
        stepOnce();
        m_stepAccumulator -= step;
        ++steps;
    }
    // Drop whatever could not be simulated within the budget instead of carrying a backlog that
    // would make the sim run in slow motion and never catch up.
    if (m_stepAccumulator > step)
    {
        m_stepAccumulator = 0.0f;
    }
}

float FluidDemo::meanDensityRatio() const
{
    const auto &densities = m_solver.densities();
    if (densities.empty() || m_solver.config().restDensity <= 0.0f)
    {
        return 0.0f;
    }
    double total = 0.0;
    for (const float density : densities)
    {
        total += density;
    }
    return static_cast<float>(total / densities.size()) / m_solver.config().restDensity;
}

std::vector<OverlayInstance> FluidDemo::buildOverlayInstances() const
{
    const auto &positions = m_solver.positions();

    std::vector<OverlayInstance> instances;
    instances.reserve(positions.size() + 1);

    // OverlayMesh::sphere() is a unit-radius sphere, so scale by the radius rather than diameter.
    const float radius = m_config.particleSpacing * m_config.particleRenderScale * 0.5f;
    for (const glm::vec3 &position : positions)
    {
        instances.push_back(OverlayInstance{
            .primitive = OverlayPrimitive::Sphere,
            .position  = position,
            .scale     = glm::vec3(radius),
            .color     = m_config.particleColor,
        });
    }
    return instances;
}

void FluidDemo::onGUI()
{
    if (ImGui::Button(m_paused ? "Play" : "Pause"))
    {
        m_paused = !m_paused;
    }
    ImGui::SameLine();
    if (ImGui::Button("Step"))
    {
        stepOnce();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset"))
    {
        reset();
    }

    ImGui::Text("Particles: %zu", particleCount());
    // The headline diagnostic: a converged solve holds this near 1. Values well above mean the
    // fluid is being compressed, so the iteration count or spacing needs attention.
    ImGui::Text("Mean density / rest: %.3f", static_cast<double>(meanDensityRatio()));

    if (ImGui::DragInt("Iterations", &m_config.solverIterations, 1.0f, 0, 32))
    {
        m_config.solverIterations = std::max(m_config.solverIterations, 0);
        rebuildSolverConfig();
    }

    // Energy has to leave the system somewhere: with viscosity at zero the positional solve feeds
    // its own corrections back into velocity and a dropped block never settles.
    if (ImGui::DragFloat("Viscosity", &m_config.viscosity, 0.005f, 0.0f, 0.5f, "%.3f"))
    {
        rebuildSolverConfig();
    }
    // At zero the walls contribute no density, so particles beside one read as free surface and get
    // extruded up the corners by the denser fluid behind them.
    if (ImGui::DragFloat("Wall Density", &m_config.boundsDensity, 0.02f, 0.0f, 2.0f, "%.2f"))
    {
        rebuildSolverConfig();
    }
    if (ImGui::DragFloat("Wall Friction", &m_config.boundsFriction, 0.005f, 0.0f, 1.0f, "%.3f"))
    {
        rebuildSolverConfig();
    }

    // Spacing changes the particle count, so it only takes effect on reset.
    ImGui::DragFloat("Spacing (needs reset)", &m_config.particleSpacing, 0.001f, 0.005f, 0.5f, "%.4f");
    ImGui::DragFloat("Particle Size", &m_config.particleRenderScale, 0.01f, 0.1f, 2.0f, "%.2f");
    ImGui::ColorEdit3("Color", &m_config.particleColor.x);
}

} // namespace lr
