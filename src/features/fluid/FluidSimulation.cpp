#include "features/fluid/FluidSimulation.hpp"

#include <imgui.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace lr
{

namespace
{

std::optional<uint64_t> sourceParticleCount(const FluidSimulation::CuboidSource &source)
{
    uint64_t count = source.dimensions.x;
    for (const uint32_t dimension : {source.dimensions.y, source.dimensions.z})
    {
        if (dimension != 0 && count > std::numeric_limits<uint64_t>::max() / dimension)
        {
            return std::nullopt;
        }
        count *= dimension;
    }
    return count;
}

std::optional<uint64_t> configuredParticleCount(const FluidSimulation::Config &config)
{
    uint64_t total = 0;
    for (const FluidSimulation::CuboidSource &source : config.sources)
    {
        const std::optional<uint64_t> count = sourceParticleCount(source);
        if (!count || total > std::numeric_limits<uint64_t>::max() - *count)
        {
            return std::nullopt;
        }
        total += *count;
    }
    return total;
}

glm::vec3 sourceLast(const FluidSimulation::CuboidSource &source, float spacing)
{
    return source.origin + glm::vec3(source.dimensions - glm::uvec3(1)) * spacing;
}

bool sourcesOverlap(const FluidSimulation::CuboidSource &a, const FluidSimulation::CuboidSource &b, float spacing)
{
    const glm::vec3 aLast = sourceLast(a, spacing);
    const glm::vec3 bLast = sourceLast(b, spacing);
    // A full spacing between the nearest particle centres is valid. Anything closer would create
    // an initially interpenetrating pair across the two lattices.
    const float minimumGap = spacing - 1e-5f;
    return !(b.origin.x - aLast.x >= minimumGap || a.origin.x - bLast.x >= minimumGap ||
             b.origin.y - aLast.y >= minimumGap || a.origin.y - bLast.y >= minimumGap ||
             b.origin.z - aLast.z >= minimumGap || a.origin.z - bLast.z >= minimumGap);
}

std::string validateSources(const FluidSimulation::Config &config)
{
    if (config.sources.empty())
    {
        return "At least one water cube is required";
    }

    const std::optional<uint64_t> total = configuredParticleCount(config);
    if (!total)
    {
        return "Configured particle count exceeds the supported range";
    }
    if (*total > FluidSolverGpu::kMaxParticles)
    {
        return "Configured particle count exceeds the 250,000 particle limit";
    }
    const float     spacing  = std::max(config.particleSpacing, 1e-4f);
    const glm::vec3 insetMin = config.boundsMin + glm::vec3(spacing);
    const glm::vec3 insetMax = config.boundsMax - glm::vec3(spacing);
    for (std::size_t i = 0; i < config.sources.size(); ++i)
    {
        const FluidSimulation::CuboidSource &source = config.sources[i];
        if (glm::any(glm::equal(source.dimensions, glm::uvec3(0))))
        {
            return "Water cube " + std::to_string(i + 1) + " has a zero dimension";
        }
        const glm::vec3 last = sourceLast(source, spacing);
        if (glm::any(glm::lessThan(source.origin, insetMin)) || glm::any(glm::greaterThan(last, insetMax)))
        {
            return "Water cube " + std::to_string(i + 1) + " does not fit inside the simulation bounds";
        }
        for (std::size_t other = 0; other < i; ++other)
        {
            if (sourcesOverlap(source, config.sources[other], spacing))
            {
                return "Water cubes " + std::to_string(other + 1) + " and " + std::to_string(i + 1) +
                       " overlap";
            }
        }
    }
    return {};
}

std::optional<FluidSimulation::CuboidSource> findSpaceForCube(const FluidSimulation::Config &config,
                                                               uint32_t side)
{
    const float     spacing  = std::max(config.particleSpacing, 1e-4f);
    const glm::vec3 insetMin = config.boundsMin + glm::vec3(spacing);
    const glm::vec3 insetMax = config.boundsMax - glm::vec3(spacing);
    const glm::vec3 extent   = glm::vec3(side - 1) * spacing;
    if (glm::any(glm::greaterThan(insetMin + extent, insetMax)))
    {
        return std::nullopt;
    }

    const glm::ivec3 steps = glm::ivec3(glm::floor((insetMax - insetMin - extent) / spacing)) + 1;
    for (int y = 0; y < steps.y; ++y)
    {
        for (int x = 0; x < steps.x; ++x)
        {
            for (int z = 0; z < steps.z; ++z)
            {
                FluidSimulation::CuboidSource candidate{{side, side, side},
                                                        insetMin + glm::vec3(x, y, z) * spacing};
                const bool occupied = std::any_of(config.sources.begin(), config.sources.end(),
                                                  [&](const auto &source) {
                                                      return sourcesOverlap(candidate, source, spacing);
                                                  });
                if (!occupied)
                {
                    return candidate;
                }
            }
        }
    }
    return std::nullopt;
}

} // namespace

float FluidSimulation::Config::smoothingRadius() const
{
    return std::max(particleSpacing, 1e-4f) * std::max(smoothingRadiusScale, 1.0f);
}

float FluidSimulation::Config::particleRadius() const
{
    return std::max(particleSpacing, 1e-4f) * std::max(particleRenderScale, 0.0f) * 1.0f;
}

float FluidSimulation::unitCubeRenderScale() const
{
    const glm::vec3 extent       = glm::abs(m_config.boundsMax - m_config.boundsMin);
    const float     longestSide  = std::max({extent.x, extent.y, extent.z});
    return longestSide > 1e-6f ? 1.0f / longestSide : 1.0f;
}

FluidSolverGpu::Config FluidSimulation::solverConfig(const Config &config)
{
    FluidSolverGpu::Config result;
    result.smoothingRadius     = config.smoothingRadius();
    result.particleMass        = 1.0f;
    result.restDensity         = FluidSolverCpu::restDensityForSpacing(std::max(config.particleSpacing, 1e-4f),
                                                                       result.smoothingRadius, result.particleMass);
    // Equation (11) leaves epsilon user-specified. 100 is a conservative normalized-unit CFM
    // regularizer; unlike the paper's published constants there is no reported scene value.
    result.relaxation          = 100.0f;
    result.solverIterations    = config.solverIterations;
    result.maxSolverIterations = std::max(config.solverIterations, 16);
    result.boundsMin           = config.boundsMin;
    result.boundsMax           = config.boundsMax;
    result.boundsFriction      = config.boundsFriction;
    result.artificialPressureK = config.artificialPressureK;
    result.vorticityEpsilon    = config.vorticityEpsilon;
    result.viscosity           = config.viscosity;
    result.maxSpeedInRadiiPerSecond = 0.0f;
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
    const std::string validationError = validateSources(config);
    if (!validationError.empty())
    {
        throw std::invalid_argument(validationError);
    }

    const uint64_t count64 = *configuredParticleCount(config);

    std::vector<glm::vec3> positions;
    positions.reserve(static_cast<std::size_t>(count64));
    const float spacing = std::max(config.particleSpacing, 1e-4f);
    for (const CuboidSource &source : config.sources)
    {
        for (uint32_t x = 0; x < source.dimensions.x; ++x)
        {
            for (uint32_t y = 0; y < source.dimensions.y; ++y)
            {
                for (uint32_t z = 0; z < source.dimensions.z; ++z)
                {
                    positions.emplace_back(source.origin + glm::vec3(x, y, z) * spacing);
                }
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
    solver.relaxation       = 100.0f;
    solver.solverIterations = std::clamp(m_config.solverIterations, 0, solver.maxSolverIterations);
    solver.boundsMin        = m_config.boundsMin;
    solver.boundsMax        = m_config.boundsMax;
    solver.boundsFriction   = m_config.boundsFriction;
    solver.artificialPressureK = m_config.artificialPressureK;
    solver.vorticityEpsilon = m_config.vorticityEpsilon;
    solver.viscosity = m_config.viscosity;
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
        m_solver.reserve(static_cast<uint32_t>(source.size()));
        applyRuntimeConfig();
        m_solver.setParticles(source);
        m_solver.setPaused(wasPaused);
        m_sourcesDirty = false;
        m_sourceEditorError.clear();
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
    const std::string sourceValidationError = validateSources(m_config);
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
    ImGui::BeginDisabled(!sourceValidationError.empty());
    if (ImGui::Button(m_sourcesDirty ? "Apply & Reset" : "Reset"))
    {
        reset();
    }
    ImGui::EndDisabled();

    ImGui::Text("Particles: %u", particleCount());
    ImGui::Text("GPU steps this frame: %u", pendingSteps());

    if (ImGui::DragFloat("Spacing (needs reset)", &m_config.particleSpacing, 0.01f, 0.05f, 1.0f, "%.3f"))
    {
        m_sourcesDirty = true;
        m_sourceEditorError.clear();
    }
    ImGui::DragFloat("Particle Size", &m_config.particleRenderScale, 0.01f, 0.1f, 2.0f, "%.2f");

    if (ImGui::CollapsingHeader("Starting Water", ImGuiTreeNodeFlags_DefaultOpen))
    {
        const std::optional<uint64_t> configuredCount = configuredParticleCount(m_config);
        if (configuredCount)
        {
            const bool overLimit = *configuredCount > FluidSolverGpu::kMaxParticles;
            ImGui::TextColored(overLimit ? ImVec4(1.0f, 0.35f, 0.35f, 1.0f)
                                         : ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                               "Configured particles: %llu / 250,000",
                               static_cast<unsigned long long>(*configuredCount));
        } else
        {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Configured particle count overflow");
        }

        std::optional<std::size_t> removeIndex;
        const float spacing = std::max(m_config.particleSpacing, 1e-4f);
        const glm::vec3 availableSize = m_config.boundsMax - m_config.boundsMin - glm::vec3(2.0f * spacing);
        const glm::uvec3 maximumDimensions =
            glm::max(glm::uvec3(1), glm::uvec3(glm::floor(availableSize / spacing)) + glm::uvec3(1));

        for (std::size_t i = 0; i < m_config.sources.size(); ++i)
        {
            CuboidSource &source = m_config.sources[i];
            ImGui::PushID(static_cast<int>(i));
            ImGui::Separator();
            ImGui::Text("Water Cube %zu", i + 1);
            ImGui::SameLine();
            ImGui::BeginDisabled(m_config.sources.size() == 1);
            if (ImGui::SmallButton("Remove"))
            {
                removeIndex = i;
            }
            ImGui::EndDisabled();

            int dimensions[3]{static_cast<int>(std::min(source.dimensions.x, uint32_t(INT_MAX))),
                              static_cast<int>(std::min(source.dimensions.y, uint32_t(INT_MAX))),
                              static_cast<int>(std::min(source.dimensions.z, uint32_t(INT_MAX)))};
            if (ImGui::DragInt3("Particles XYZ", dimensions, 1.0f, 1, 10000))
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    dimensions[axis] = std::clamp(dimensions[axis], 1, static_cast<int>(maximumDimensions[axis]));
                }
                source.dimensions = glm::uvec3(dimensions[0], dimensions[1], dimensions[2]);
                const glm::vec3 minOrigin = m_config.boundsMin + glm::vec3(spacing);
                const glm::vec3 maxOrigin = m_config.boundsMax - glm::vec3(spacing) -
                                            glm::vec3(source.dimensions - glm::uvec3(1)) * spacing;
                source.origin   = glm::clamp(source.origin, minOrigin, maxOrigin);
                m_sourcesDirty = true;
                m_sourceEditorError.clear();
            }

            if (ImGui::DragFloat3("Origin", &source.origin.x, 0.1f, 0.0f, 0.0f, "%.2f"))
            {
                const glm::vec3 minOrigin = m_config.boundsMin + glm::vec3(spacing);
                const glm::vec3 maxOrigin = m_config.boundsMax - glm::vec3(spacing) -
                                            glm::vec3(source.dimensions - glm::uvec3(1)) * spacing;
                source.origin  = glm::clamp(source.origin, minOrigin, maxOrigin);
                m_sourcesDirty = true;
                m_sourceEditorError.clear();
            }
            ImGui::PopID();
        }

        if (removeIndex)
        {
            m_config.sources.erase(m_config.sources.begin() + static_cast<std::ptrdiff_t>(*removeIndex));
            m_sourcesDirty = true;
            m_sourceEditorError.clear();
        }

        const bool canAddCube = validateSources(m_config).empty();
        ImGui::BeginDisabled(!canAddCube);
        if (ImGui::Button("Add Cube"))
        {
            std::optional<CuboidSource> newCube;
            // Search only on click: a completely full domain can have hundreds of thousands of
            // candidate lattice positions, so doing this every rendered frame would stall the UI.
            for (uint32_t side = 10; side > 0 && !newCube; --side)
            {
                newCube = findSpaceForCube(m_config, side);
            }
            if (newCube)
            {
                m_config.sources.push_back(*newCube);
                m_sourcesDirty = true;
                m_sourceEditorError.clear();
            } else
            {
                m_sourceEditorError = "No valid empty space is available inside the simulation bounds";
            }
        }
        ImGui::EndDisabled();
        if (!m_sourceEditorError.empty())
        {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", m_sourceEditorError.c_str());
        }

        const std::string updatedValidationError = validateSources(m_config);
        if (!updatedValidationError.empty())
        {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", updatedValidationError.c_str());
        } else if (m_sourcesDirty)
        {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.25f, 1.0f), "Apply & Reset to use this layout");
        }
    }

    auto &solver = m_solver.config();
    if (ImGui::DragInt("Iterations", &m_config.solverIterations, 1.0f, 0, solver.maxSolverIterations))
    {
        m_config.solverIterations = std::clamp(m_config.solverIterations, 0, solver.maxSolverIterations);
        solver.solverIterations   = m_config.solverIterations;
        m_solver.updateParams();
    }
    if (ImGui::DragFloat("XSPH scale", &m_config.viscosity, 0.001f, 0.0f, 0.5f, "%.3f"))
    {
        solver.viscosity = m_config.viscosity;
        m_solver.updateParams();
    }
    if (ImGui::DragFloat("Artificial pressure scale", &m_config.artificialPressureK, 0.005f, 0.0f, 1.0f, "%.3f"))
    {
        solver.artificialPressureK = m_config.artificialPressureK;
        m_solver.updateParams();
    }
    if (ImGui::DragFloat("Vorticity", &m_config.vorticityEpsilon, 0.001f, 0.0f, 10.0f, "%.3f"))
    {
        solver.vorticityEpsilon = m_config.vorticityEpsilon;
        m_solver.updateParams();
    }
    if (ImGui::DragFloat("Wall Friction", &m_config.boundsFriction, 0.005f, 0.0f, 1.0f, "%.3f"))
    {
        solver.boundsFriction = m_config.boundsFriction;
        m_solver.updateParams();
    }

    ImGui::ColorEdit3("Color", &m_config.particleColor.x);
    if (!m_error.empty())
    {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", m_error.c_str());
    }
}

} // namespace lr
