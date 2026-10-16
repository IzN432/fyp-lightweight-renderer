#include "features/fluid/FluidSimulation.hpp"

#include "core/passes/overlaylines/OverlayLine.hpp"

#include <glm/ext/matrix_transform.hpp>
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

constexpr OverlayLineStyle kBoundsStyle{
    .color           = {0.45f, 0.70f, 0.95f},
    .visibleOpacity  = 0.9f,
    .occludedOpacity = 0.2f,
};

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
                return "Water cubes " + std::to_string(other + 1) + " and " + std::to_string(i + 1) + " overlap";
            }
        }
    }
    return {};
}

std::optional<FluidSimulation::CuboidSource> findSpaceForCube(const FluidSimulation::Config &config, uint32_t side)
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
                FluidSimulation::CuboidSource candidate{{side, side, side}, insetMin + glm::vec3(x, y, z) * spacing};
                const bool                    occupied =
                    std::any_of(config.sources.begin(), config.sources.end(), [&](const auto &source) {
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

// Pulls every water cube back inside the inset domain. The bounds are freely editable at runtime,
// so shrinking the box would otherwise leave sources hanging outside it -- a state validateSources
// rejects, which disables Reset and strands the editor. Cubes larger than the new domain cannot be
// made to fit by moving them; those are left to the validation message. Returns true only when a
// cube actually moved, so a bounds nudge that disturbs nothing does not ask for a respawn.
bool clampSourcesToBounds(FluidSimulation::Config &config)
{
    const float     spacing  = std::max(config.particleSpacing, 1e-4f);
    const glm::vec3 insetMin = config.boundsMin + glm::vec3(spacing);
    const glm::vec3 insetMax = config.boundsMax - glm::vec3(spacing);
    bool            moved    = false;
    for (FluidSimulation::CuboidSource &source : config.sources)
    {
        const glm::vec3 extent    = glm::vec3(source.dimensions - glm::uvec3(1)) * spacing;
        const glm::vec3 maxOrigin = glm::max(insetMax - extent, insetMin);
        const glm::vec3 clamped   = glm::clamp(source.origin, insetMin, maxOrigin);
        if (clamped != source.origin)
        {
            source.origin = clamped;
            moved         = true;
        }
    }
    return moved;
}

} // namespace

FluidSolverGpu::Config FluidSimulation::solverConfig(const Config &config)
{
    FluidSolverGpu::Config result;
    result.smoothingRadius = config.smoothingRadius();
    result.particleMass    = 1.0f;
    result.restDensity     = FluidSolverCpu::restDensityForSpacing(std::max(config.particleSpacing, 1e-4f),
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
    result.maxDisplacementInRadiiPerStep =
        std::max(config.maxDisplacementScale, 0.0f) * std::max(config.particleSpacing, 1e-4f) / result.smoothingRadius;
    result.maxParticles     = config.maxParticles;
    result.fixedDeltaTime   = config.fixedDeltaTime;
    result.maxStepsPerFrame = config.maxStepsPerFrame;
    result.startPaused      = config.startPaused;
    result.particleRadius   = config.particleRadius();
    result.particleColor    = config.particleColor;
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

FluidSimulation::FluidSimulation(Config config, ResourceRegistry &resources, OfflineFluidSimulationStore &offlineStore)
    : m_config(std::move(config)), m_solver(solverConfig(m_config), resources), m_offlineStore(&offlineStore)
{
    reset();
}

void FluidSimulation::build(FrameGraph &frameGraph) { m_solver.build(frameGraph); }

void FluidSimulation::setTarget(FluidComponent *component, const glm::vec3 &worldPosition)
{
    m_worldPosition = worldPosition;
    if (component == m_target)
    {
        // A paste or undo can replace all settings without going through this panel.
        if (component && component->revision() != m_targetRevision)
        {
            m_targetRevision = component->revision();
            if (component->isLive())
            {
                m_config = component->config();
                m_solver.setPublishEnabled(true);
                reset();
                m_solver.setPaused(m_config.startPaused);
            } else
            {
                m_bakedConfig = component->bakedConfig();
                resetBaked();
            }
        }
        return;
    }

    m_target              = component;
    m_singleStepRequested = false;
    m_sourcesDirty        = false;
    m_sourceEditorError.clear();
    m_error.clear();
    if (!m_target)
    {
        m_bakedParticleCount = 0;
        m_solver.setPublishEnabled(false);
        return;
    }

    m_targetRevision = m_target->revision();
    if (m_target->isLive())
    {
        m_config = m_target->config();
        m_solver.setPublishEnabled(true);
        reset();
        m_solver.setPaused(m_config.startPaused);
    } else
    {
        m_bakedConfig = m_target->bakedConfig();
        resetBaked();
    }
}

void FluidSimulation::applyRuntimeConfig()
{
    auto &solver               = m_solver.config();
    solver.smoothingRadius     = m_config.smoothingRadius();
    solver.restDensity         = FluidSolverCpu::restDensityForSpacing(std::max(m_config.particleSpacing, 1e-4f),
                                                                       solver.smoothingRadius, solver.particleMass);
    solver.relaxation          = 100.0f;
    solver.solverIterations    = std::clamp(m_config.solverIterations, 0, solver.maxSolverIterations);
    solver.boundsMin           = m_config.boundsMin;
    solver.boundsMax           = m_config.boundsMax;
    solver.boundsFriction      = m_config.boundsFriction;
    solver.artificialPressureK = m_config.artificialPressureK;
    solver.vorticityEpsilon    = m_config.vorticityEpsilon;
    solver.viscosity           = m_config.viscosity;
    solver.maxDisplacementInRadiiPerStep = std::max(m_config.maxDisplacementScale, 0.0f) *
                                           std::max(m_config.particleSpacing, 1e-4f) / solver.smoothingRadius;
    solver.particleRadius                = m_config.particleRadius();
    solver.particleColor                 = m_config.particleColor;
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

void FluidSimulation::resetBaked()
{
    m_solver.setPaused(true);
    m_solver.update(0.0f);
    m_solver.setPublishEnabled(false);
    m_playbackTime       = 0.0f;
    m_bakedFrame         = std::numeric_limits<std::size_t>::max();
    m_bakedParticleCount = 0;
    m_bakedPaused        = m_bakedConfig.startPaused;
    if (!m_offlineStore || !m_offlineStore->contains(m_bakedConfig.simulation))
    {
        m_error = "Baked fluid simulation handle is invalid";
        return;
    }
    const OfflineFluidSimulation &simulation = m_offlineStore->get(m_bakedConfig.simulation);
    if (simulation.frames().empty())
    {
        m_error = "Baked fluid simulation has no frames";
        return;
    }
    uploadBakedFrame(0);
    m_error.clear();
}

float FluidSimulation::resolvedBakedParticleRadius() const
{
    if (m_bakedConfig.particleRadius > 0.0f)
    {
        return m_bakedConfig.particleRadius;
    }
    if (!m_offlineStore || !m_offlineStore->contains(m_bakedConfig.simulation))
    {
        return 0.025f;
    }
    return m_offlineStore->get(m_bakedConfig.simulation).defaultParticleRadius();
}

void FluidSimulation::uploadBakedFrame(std::size_t frameIndex)
{
    const OfflineFluidSimulation &simulation = m_offlineStore->get(m_bakedConfig.simulation);
    if (frameIndex >= simulation.frames().size())
    {
        throw std::out_of_range("Baked fluid frame is out of range");
    }
    const std::span<const glm::vec3> positions = simulation.framePositions(frameIndex);
    m_bakedParticles.clear();
    m_bakedParticles.reserve(positions.size());
    for (const glm::vec3 &position : positions)
    {
        m_bakedParticles.push_back(
            makeParticleGpuData(position, resolvedBakedParticleRadius(), m_bakedConfig.particleColor));
    }
    m_solver.setPresentationParticles(m_bakedParticles);
    m_bakedParticleCount = static_cast<uint32_t>(m_bakedParticles.size());
    m_bakedFrame         = frameIndex;
}

void FluidSimulation::update(float frameDeltaTime)
{
    if (!m_target)
    {
        return;
    }

    if (m_target->isBaked())
    {
        const OfflineFluidSimulation &simulation = m_offlineStore->get(m_bakedConfig.simulation);
        if (!m_bakedPaused && frameDeltaTime > 0.0f)
        {
            m_playbackTime += frameDeltaTime * std::max(m_bakedConfig.playbackSpeed, 0.0f);
        }
        const float duration = simulation.durationSeconds();
        if (duration > 0.0f)
        {
            if (m_bakedConfig.loop)
            {
                m_playbackTime = std::fmod(m_playbackTime, duration);
            } else
            {
                m_playbackTime = std::min(m_playbackTime, std::nextafter(duration, 0.0f));
            }
        }
        const auto found = std::upper_bound(simulation.frames().begin(), simulation.frames().end(), m_playbackTime,
                                            [](float seconds, const OfflineFluidSimulation::Frame &frame) {
                                                return seconds < frame.seconds;
                                            });
        const std::size_t frame = found == simulation.frames().begin()
                                      ? 0
                                      : static_cast<std::size_t>(std::distance(simulation.frames().begin(), found) - 1);
        if (frame != m_bakedFrame)
        {
            uploadBakedFrame(frame);
        }
        return;
    }

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

uint32_t FluidSimulation::particleCount() const
{
    if (!m_target)
    {
        return 0;
    }
    return m_target->isBaked() ? m_bakedParticleCount : m_solver.particleCount();
}

uint32_t FluidSimulation::pendingSteps() const { return m_target && m_target->isLive() ? m_solver.pendingSteps() : 0; }

float FluidSimulation::particleRadius() const
{
    return m_target && m_target->isBaked() ? resolvedBakedParticleRadius() : m_config.particleRadius();
}

const glm::vec3 &FluidSimulation::particleColor() const
{
    return m_target && m_target->isBaked() ? m_bakedConfig.particleColor : m_config.particleColor;
}

float FluidSimulation::renderScale() const
{
    return m_target && m_target->isBaked() ? m_bakedConfig.renderScale : m_config.renderScale;
}

void FluidSimulation::appendOverlayLines(OverlayLineBuilder &builder, float visualScale) const
{
    if (!m_target || m_target->isBaked() || !m_config.showBounds)
    {
        return;
    }

    const float     scale  = std::max(visualScale, 0.0f);
    const glm::vec3 centre = m_worldPosition + (m_config.boundsMin + m_config.boundsMax) * (0.5f * scale);
    const glm::vec3 size   = (m_config.boundsMax - m_config.boundsMin) * scale;
    const glm::mat4 model  = glm::scale(glm::translate(glm::mat4(1.0f), centre), size);
    builder.addPrimitive(primitives::cube.outline, model, kBoundsStyle);
}

void FluidSimulation::onGUI()
{
    if (!m_target)
    {
        return;
    }

    if (m_target->isBaked())
    {
        const OfflineFluidSimulation     &simulation = m_offlineStore->get(m_bakedConfig.simulation);
        const FluidComponent::BakedConfig before     = m_bakedConfig;
        if (ImGui::Button(m_bakedPaused ? "Play" : "Pause"))
        {
            m_bakedPaused = !m_bakedPaused;
        }
        ImGui::SameLine();
        if (ImGui::Button("Restart"))
        {
            m_playbackTime = 0.0f;
            uploadBakedFrame(0);
        }
        ImGui::Text("Baked clip: %s", simulation.name().c_str());
        ImGui::Text("Frame: %zu / %zu", m_bakedFrame + 1, simulation.frames().size());
        ImGui::Text("Particles: %u", m_bakedParticleCount);
        ImGui::DragFloat("Playback Speed", &m_bakedConfig.playbackSpeed, 0.01f, 0.0f, 8.0f, "%.2fx");
        ImGui::Checkbox("Loop", &m_bakedConfig.loop);
        ImGui::DragFloat("Particle Size", &m_bakedConfig.particleRadius, 0.001f, 0.0f, 10.0f, "%.4f");
        ImGui::TextDisabled("0 uses imported radius: %.4f", simulation.defaultParticleRadius());
        ImGui::DragFloat("Render Scale", &m_bakedConfig.renderScale, 0.001f, 0.001f, 100.0f, "%.4f");
        ImGui::ColorEdit3("Color", &m_bakedConfig.particleColor.x);
        if (!(before == m_bakedConfig))
        {
            m_target->setBakedConfig(m_bakedConfig);
            m_targetRevision = m_target->revision();
            uploadBakedFrame(m_bakedFrame);
        }
        if (!m_error.empty())
        {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", m_error.c_str());
        }
        return;
    }

    const Config      before                = m_config;
    const std::string sourceValidationError = validateSources(m_config);
    const bool        paused                = m_solver.paused();
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
    ImGui::DragFloat("Simulation Render Scale", &m_config.renderScale, 0.001f, 0.001f, 10.0f, "%.4f");

    if (ImGui::CollapsingHeader("Starting Water", ImGuiTreeNodeFlags_DefaultOpen))
    {
        const std::optional<uint64_t> configuredCount = configuredParticleCount(m_config);
        if (configuredCount)
        {
            const bool overLimit = *configuredCount > FluidSolverGpu::kMaxParticles;
            ImGui::TextColored(overLimit ? ImVec4(1.0f, 0.35f, 0.35f, 1.0f) : ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                               "Configured particles: %llu / 250,000",
                               static_cast<unsigned long long>(*configuredCount));
        } else
        {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Configured particle count overflow");
        }

        std::optional<std::size_t> removeIndex;
        const float                spacing       = std::max(m_config.particleSpacing, 1e-4f);
        const glm::vec3            availableSize = m_config.boundsMax - m_config.boundsMin - glm::vec3(2.0f * spacing);
        const glm::uvec3           maximumDimensions =
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
                source.dimensions         = glm::uvec3(dimensions[0], dimensions[1], dimensions[2]);
                const glm::vec3 minOrigin = m_config.boundsMin + glm::vec3(spacing);
                const glm::vec3 maxOrigin =
                    m_config.boundsMax - glm::vec3(spacing) - glm::vec3(source.dimensions - glm::uvec3(1)) * spacing;
                source.origin  = glm::clamp(source.origin, minOrigin, maxOrigin);
                m_sourcesDirty = true;
                m_sourceEditorError.clear();
            }

            if (ImGui::DragFloat3("Origin", &source.origin.x, 0.1f, 0.0f, 0.0f, "%.2f"))
            {
                const glm::vec3 minOrigin = m_config.boundsMin + glm::vec3(spacing);
                const glm::vec3 maxOrigin =
                    m_config.boundsMax - glm::vec3(spacing) - glm::vec3(source.dimensions - glm::uvec3(1)) * spacing;
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
    if (ImGui::CollapsingHeader("Domain"))
    {
        // Nothing in the solver is sized by the domain: the neighbour grid hashes world cells into
        // a fixed bucket table, so the box only clamps positions and seeds the spawn validation.
        // That makes it safe to resize while the sim is running -- no reset, no reallocation. On
        // screen the domain is pure presentation too: apparent size comes from Simulation Render
        // Scale alone, so resizing the walls moves them without rescaling the fluid.
        glm::vec3  centre       = (m_config.boundsMin + m_config.boundsMax) * 0.5f;
        glm::vec3  size         = m_config.boundsMax - m_config.boundsMin;
        const bool centreEdited = ImGui::DragFloat3("Domain Centre", &centre.x, 0.1f, 0.0f, 0.0f, "%.2f");
        const bool sizeEdited   = ImGui::DragFloat3("Domain Size", &size.x, 0.1f, 0.0f, 0.0f, "%.2f");
        if (centreEdited || sizeEdited)
        {
            // A degenerate or inverted axis would make insetMin exceed insetMax and leave the
            // clamp in fluid_apply.comp undefined, so each extent keeps a floor of one spacing.
            const float minimumExtent = 2.0f * std::max(m_config.particleSpacing, 1e-4f);
            size                      = glm::max(size, glm::vec3(minimumExtent));
            m_config.boundsMin        = centre - size * 0.5f;
            m_config.boundsMax        = centre + size * 0.5f;
            solver.boundsMin          = m_config.boundsMin;
            solver.boundsMax          = m_config.boundsMax;
            m_solver.updateParams();
            if (clampSourcesToBounds(m_config))
            {
                m_sourcesDirty = true;
                m_sourceEditorError.clear();
            }
        }
        ImGui::TextDisabled("Min %.2f, %.2f, %.2f", m_config.boundsMin.x, m_config.boundsMin.y, m_config.boundsMin.z);
        ImGui::TextDisabled("Max %.2f, %.2f, %.2f", m_config.boundsMax.x, m_config.boundsMax.y, m_config.boundsMax.z);
        ImGui::TextDisabled("Walls move live; Reset only to respawn the water.");
    }
    ImGui::Checkbox("Show Bounds", &m_config.showBounds);

    ImGui::ColorEdit3("Color", &m_config.particleColor.x);
    if (!m_error.empty())
    {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", m_error.c_str());
    }

    if (!(before == m_config))
    {
        m_target->setConfig(m_config);
        m_targetRevision = m_target->revision();
    }
}

} // namespace lr
