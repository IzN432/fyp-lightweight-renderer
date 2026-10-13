#pragma once

#include "features/fluid/FluidComponent.hpp"
#include "features/fluid/FluidSolverGpu.hpp"
#include "features/fluid/OfflineFluidSimulationStore.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <limits>
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
    using CuboidSource = FluidComponent::CuboidSource;
    using Config       = FluidComponent::Config;

    FluidSimulation(Config config, ResourceRegistry &resources, OfflineFluidSimulationStore &offlineStore);

    void build(FrameGraph &frameGraph);
    // Binds the one simulation the shared GPU runtime displays. Passing null hides it and stops it.
    void setTarget(FluidComponent *component, const glm::vec3 &worldPosition = glm::vec3(0.0f));
    void update(float frameDeltaTime);
    void reset();
    void onGUI();
    // Appends the presentation-scaled domain outline when Show Bounds is enabled.
    void appendOverlayLines(OverlayLineBuilder &builder, float visualScale) const;

    uint32_t particleCount() const;
    uint32_t capacity() const { return m_solver.config().maxParticles; }
    uint32_t pendingSteps() const;
    float    particleRadius() const;
    // Uniform presentation scale whose largest domain dimension is exactly one world unit.
    float unitCubeRenderScale() const;

    const glm::vec3   &particleColor() const;
    const glm::vec3   &boundsMin() const { return m_config.boundsMin; }
    const glm::vec3   &boundsMax() const { return m_config.boundsMax; }
    bool               showBounds() const { return m_config.showBounds; }
    const glm::vec3   &worldPosition() const { return m_worldPosition; }
    float              renderScale() const;
    const std::string &particleBufferName() const { return m_solver.config().particleBufferResourceName; }

private:
    static FluidSolverGpu::Config solverConfig(const Config &config);
    static std::vector<glm::vec3> makeSource(const Config &config);
    void                          applyRuntimeConfig();
    void                          resetBaked();
    void                          uploadBakedFrame(std::size_t frameIndex);
    float                         resolvedBakedParticleRadius() const;

    Config                       m_config;
    FluidSolverGpu               m_solver;
    OfflineFluidSimulationStore *m_offlineStore   = nullptr;
    FluidComponent              *m_target         = nullptr;
    uint64_t                     m_targetRevision = 0;
    glm::vec3                    m_worldPosition{0.0f};
    bool                         m_singleStepRequested = false;
    bool                         m_sourcesDirty        = false;
    std::string                  m_sourceEditorError;
    std::string                  m_error;
    FluidComponent::BakedConfig  m_bakedConfig;
    float                        m_playbackTime       = 0.0f;
    std::size_t                  m_bakedFrame         = std::numeric_limits<std::size_t>::max();
    uint32_t                     m_bakedParticleCount = 0;
    bool                         m_bakedPaused        = false;
    std::vector<ParticleGpuData> m_bakedParticles;
};

} // namespace lr
