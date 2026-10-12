#include "features/fluid/FluidDemo.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lr
{

FluidDemo::FluidDemo(Config config) : m_config(config), m_paused(config.startPaused)
{
    reset();
    m_worker = std::jthread([this](std::stop_token stop) { runWorker(stop); });
}

FluidDemo::~FluidDemo()
{
    m_worker.request_stop();
    m_workAvailable.notify_one();
    if (m_worker.joinable()) m_worker.join();
}

bool FluidDemo::collectUpdate(bool wait)
{
    std::unique_lock lock(m_workerMutex);
    if (wait)
    {
        m_workerIdle.wait(lock, [this] { return m_pendingSteps == 0 && !m_workerBusy; });
    } else if (m_pendingSteps > 0 || m_workerBusy)
    {
        return false;
    }

    const bool         snapshotReady = m_snapshotReady;
    std::exception_ptr error         = m_workerError;
    m_snapshotReady                   = false;
    m_workerError                     = nullptr;
    lock.unlock();

    if (error) std::rethrow_exception(error);
    if (snapshotReady) refreshSnapshot();
    return true;
}

void FluidDemo::dispatchSteps(int steps, float deltaTime)
{
    {
        std::lock_guard lock(m_workerMutex);
        m_pendingSteps     = steps;
        m_pendingDeltaTime = deltaTime;
    }
    m_workAvailable.notify_one();
}

void FluidDemo::runWorker(std::stop_token stop)
{
    while (!stop.stop_requested())
    {
        int   steps;
        float deltaTime;
        {
            std::unique_lock lock(m_workerMutex);
            m_workAvailable.wait(lock, [&] { return stop.stop_requested() || m_pendingSteps > 0; });
            if (stop.stop_requested()) return;
            steps              = m_pendingSteps;
            deltaTime          = m_pendingDeltaTime;
            m_pendingSteps     = 0;
            m_workerBusy       = true;
        }

        try
        {
            for (int i = 0; i < steps; ++i) m_solver.step(deltaTime);
        } catch (...)
        {
            std::lock_guard lock(m_workerMutex);
            m_workerError = std::current_exception();
        }

        {
            std::lock_guard lock(m_workerMutex);
            m_workerBusy    = false;
            m_snapshotReady = true;
        }
        m_workerIdle.notify_all();
    }
}

void FluidDemo::refreshSnapshot()
{
    m_positions = m_solver.positions();
    const auto &densities = m_solver.densities();
    if (densities.empty() || m_solver.config().restDensity <= 0.0f)
    {
        m_meanDensityRatio = 0.0f;
        return;
    }
    double total = 0.0;
    for (const float density : densities) total += density;
    m_meanDensityRatio = static_cast<float>(total / densities.size()) / m_solver.config().restDensity;
}

void FluidDemo::rebuildSolverConfig()
{
    collectUpdate(true);
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
    // Equations (13) and (17): use the configured coefficients literally.
    solverConfig.artificialPressureK = m_config.artificialPressureK;
    solverConfig.vorticityEpsilon    = m_config.vorticityEpsilon;
    solverConfig.viscosity = m_config.viscosity;
    solverConfig.boundsFriction   = m_config.boundsFriction;
    solverConfig.boundsMin        = m_config.boundsMin;
    solverConfig.boundsMax        = m_config.boundsMax;
    m_solver.setConfig(solverConfig);
    refreshSnapshot();
}

void FluidDemo::reset()
{
    rebuildSolverConfig();

    const float     spacing = std::max(m_config.particleSpacing, 1e-4f);
    if (glm::any(glm::equal(m_config.sourceDimensions, glm::uvec3(0))))
    {
        throw std::invalid_argument("Fluid source dimensions must be greater than zero");
    }
    const std::size_t count = static_cast<std::size_t>(m_config.sourceDimensions.x) *
                              static_cast<std::size_t>(m_config.sourceDimensions.y) *
                              static_cast<std::size_t>(m_config.sourceDimensions.z);
    if (count > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument("Fluid source particle count exceeds the supported range");
    }

    // Inset by one spacing so no particle starts exactly on a wall, where the boundary clamp would
    // fight the density solve on the very first step.
    const glm::vec3 min = m_config.boundsMin + glm::vec3(spacing);
    const glm::vec3 max = min + glm::vec3(m_config.sourceDimensions - glm::uvec3(1)) * spacing;
    if (glm::any(glm::greaterThan(max, m_config.boundsMax - glm::vec3(spacing))))
    {
        throw std::invalid_argument("Fluid source lattice does not fit inside the simulation bounds");
    }

    m_solver.setParticles(FluidSolverCpu::makeParticleBlock(min, max, spacing));
    if (m_solver.particleCount() != count)
    {
        throw std::runtime_error("Fluid source lattice produced an unexpected particle count");
    }
    m_stepAccumulator = 0.0f;
    m_paused          = true;
    refreshSnapshot();
}

void FluidDemo::stepOnce()
{
    collectUpdate(true);
    const float step = std::max(m_config.fixedDeltaTime, 1e-5f);
    dispatchSteps(1, step);
}

void FluidDemo::update(float frameDeltaTime)
{
    const bool workerReady = collectUpdate(false);
    if (m_paused || frameDeltaTime <= 0.0f)
    {
        return;
    }

    const float step = std::max(m_config.fixedDeltaTime, 1e-5f);
    m_stepAccumulator += frameDeltaTime;
    if (!workerReady) return;

    const int maxSteps = std::max(m_config.maxStepsPerFrame, 1);
    int       steps    = 0;
    while (m_stepAccumulator >= step && steps < maxSteps)
    {
        m_stepAccumulator -= step;
        ++steps;
    }
    // Drop whatever could not be simulated within the budget instead of carrying a backlog that
    // would make the sim run in slow motion and never catch up.
    if (m_stepAccumulator > step)
    {
        m_stepAccumulator = 0.0f;
    }
    if (steps > 0)
    {
        dispatchSteps(steps, step);
    }
}

float FluidDemo::meanDensityRatio() const
{
    return m_meanDensityRatio;
}

std::vector<OverlayInstance> FluidDemo::buildOverlayInstances() const
{
    std::vector<OverlayInstance> instances;
    instances.reserve(m_positions.size() + 1);

    // OverlayMesh::sphere() is a unit-radius sphere, so scale by the radius rather than diameter.
    const float radius = m_config.particleSpacing * m_config.particleRenderScale * 0.5f;
    for (const glm::vec3 &position : m_positions)
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
    if (ImGui::DragFloat("XSPH scale", &m_config.viscosity, 0.001f, 0.0f, 0.5f, "%.3f"))
    {
        rebuildSolverConfig();
    }
    if (ImGui::DragFloat("Artificial pressure scale", &m_config.artificialPressureK, 0.005f, 0.0f, 1.0f, "%.3f"))
    {
        rebuildSolverConfig();
    }
    if (ImGui::DragFloat("Vorticity", &m_config.vorticityEpsilon, 0.001f, 0.0f, 10.0f, "%.3f"))
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
