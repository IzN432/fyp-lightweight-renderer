#include "features/fluid/FluidSimulation.hpp"

#include <imgui.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lr
{

float FluidSimulation::Config::smoothingRadius() const
{
    return std::max(particleSpacing, 1e-4f) * std::max(smoothingRadiusScale, 1.0f);
}

float FluidSimulation::Config::particleRadius() const
{
    return std::max(particleSpacing, 1e-4f) * std::max(particleRenderScale, 0.0f) * 0.5f;
}

FluidSolverGpu::Config FluidSimulation::solverConfig(const Config &config)
{
    FluidSolverGpu::Config result;
    result.smoothingRadius     = config.smoothingRadius();
    result.particleMass        = 1.0f;
    result.restDensity         = FluidSolverCpu::restDensityForSpacing(std::max(config.particleSpacing, 1e-4f),
                                                                       result.smoothingRadius, result.particleMass);
    result.relaxation          = result.restDensity * 0.01f;
    result.solverIterations    = config.solverIterations;
    result.maxSolverIterations = std::max(config.solverIterations, 16);
    result.boundsMin           = config.boundsMin;
    result.boundsMax           = config.boundsMax;
    result.boundsFriction      = config.boundsFriction;
    result.viscosity           = config.viscosity;
    result.maxParticles        = config.maxParticles;
    result.fixedDeltaTime      = config.fixedDeltaTime;
    result.maxStepsPerFrame    = config.maxStepsPerFrame;
    result.startPaused         = config.startPaused;
    result.particleRadius      = config.particleRadius();
    result.particleColor       = config.particleColor;
    return result;
}

std::vector<glm::vec3> FluidSimulation::makeSource(const Config &config)
{
    if (glm::any(glm::equal(config.sourceDimensions, glm::uvec3(0))))
    {
        throw std::invalid_argument("Fluid source dimensions must be greater than zero");
    }
    const uint64_t count64 =
        uint64_t(config.sourceDimensions.x) * config.sourceDimensions.y * config.sourceDimensions.z;
    if (count64 > config.maxParticles || count64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument("Fluid source particle count exceeds configured GPU capacity");
    }

    const float     spacing = std::max(config.particleSpacing, 1e-4f);
    const glm::vec3 first   = config.boundsMin + glm::vec3(spacing);
    const glm::vec3 last    = first + glm::vec3(config.sourceDimensions - glm::uvec3(1)) * spacing;
    if (glm::any(glm::greaterThan(last, config.boundsMax - glm::vec3(spacing))))
    {
        throw std::invalid_argument("Fluid source lattice does not fit inside the simulation bounds");
    }

    std::vector<glm::vec3> positions;
    positions.reserve(static_cast<std::size_t>(count64));
    for (uint32_t x = 0; x < config.sourceDimensions.x; ++x)
    {
        for (uint32_t y = 0; y < config.sourceDimensions.y; ++y)
        {
            for (uint32_t z = 0; z < config.sourceDimensions.z; ++z)
            {
                positions.emplace_back(first + glm::vec3(x, y, z) * spacing);
            }
        }
    }
    return positions;
}

FluidSimulation::FluidSimulation(Config config, ResourceRegistry &resources)
    : m_config(std::move(config)), m_solver(solverConfig(m_config), resources)
{
    reset();
}

void FluidSimulation::build(FrameGraph &frameGraph) { m_solver.build(frameGraph); }

void FluidSimulation::applyRuntimeConfig()
{
    auto &solver            = m_solver.config();
    solver.smoothingRadius  = m_config.smoothingRadius();
    solver.restDensity      = FluidSolverCpu::restDensityForSpacing(std::max(m_config.particleSpacing, 1e-4f),
                                                                    solver.smoothingRadius, solver.particleMass);
    solver.relaxation       = solver.restDensity * 0.01f;
    solver.solverIterations = std::clamp(m_config.solverIterations, 0, solver.maxSolverIterations);
    solver.boundsMin        = m_config.boundsMin;
    solver.boundsMax        = m_config.boundsMax;
    solver.boundsFriction   = m_config.boundsFriction;
    solver.viscosity        = m_config.viscosity;
    solver.particleRadius   = m_config.particleRadius();
    solver.particleColor    = m_config.particleColor;
    m_solver.setFixedDeltaTime(m_config.fixedDeltaTime);
    m_solver.updateParams();
}

void FluidSimulation::reset()
{
    try
    {
        std::vector<glm::vec3> source    = makeSource(m_config);
        const bool             wasPaused = m_solver.paused();
        m_singleStepRequested            = false;
        applyRuntimeConfig();
        m_solver.setParticles(source);
        m_solver.setPaused(wasPaused);
        m_error.clear();
    } catch (const std::exception &e)
    {
        m_error = e.what();
    }
}

void FluidSimulation::update(float frameDeltaTime)
{
    if (m_singleStepRequested)
    {
        m_solver.stepOnce();
        m_singleStepRequested = false;
    } else
    {
        m_solver.update(frameDeltaTime);
    }

    // Appearance is consumed by the publish pass, so these take effect without a buffer upload.
    m_solver.config().particleRadius = m_config.particleRadius();
    m_solver.config().particleColor  = m_config.particleColor;
}

void FluidSimulation::onGUI()
{
    const bool paused = m_solver.paused();
    if (ImGui::Button(paused ? "Play" : "Pause"))
    {
        m_solver.setPaused(!paused);
    }
    ImGui::SameLine();
    if (ImGui::Button("Step"))
    {
        m_singleStepRequested = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset"))
    {
        reset();
    }

    ImGui::Text("Particles: %u / %u", particleCount(), capacity());
    ImGui::Text("GPU steps this frame: %u", pendingSteps());

    auto &solver = m_solver.config();
    if (ImGui::DragInt("Iterations", &m_config.solverIterations, 1.0f, 0, solver.maxSolverIterations))
    {
        m_config.solverIterations = std::clamp(m_config.solverIterations, 0, solver.maxSolverIterations);
        solver.solverIterations   = m_config.solverIterations;
        m_solver.updateParams();
    }
    if (ImGui::DragFloat("Viscosity", &m_config.viscosity, 0.005f, 0.0f, 0.5f, "%.3f"))
    {
        solver.viscosity = m_config.viscosity;
        m_solver.updateParams();
    }
    if (ImGui::DragFloat("Wall Friction", &m_config.boundsFriction, 0.005f, 0.0f, 1.0f, "%.3f"))
    {
        solver.boundsFriction = m_config.boundsFriction;
        m_solver.updateParams();
    }

    ImGui::DragFloat("Spacing (needs reset)", &m_config.particleSpacing, 0.0001f, 0.005f, 0.015f, "%.4f");
    ImGui::DragFloat("Particle Size", &m_config.particleRenderScale, 0.01f, 0.1f, 2.0f, "%.2f");
    ImGui::ColorEdit3("Color", &m_config.particleColor.x);
    if (!m_error.empty())
    {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", m_error.c_str());
    }
}

} // namespace lr
