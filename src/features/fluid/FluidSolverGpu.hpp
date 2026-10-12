#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/framegraph/PassBuilder.hpp"
#include "core/framegraph/ResourceRegistry.hpp"
#include "features/fluid/FluidSolverCpu.hpp"

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace lr
{

// Position Based Fluids on the GPU: the same solve as FluidSolverCpu, as a chain of compute
// passes over buffers that never leave device memory.
//
// The CPU solver stays the reference. Every stage here is a transcription of the correspondingly
// named method there, down to the kernel constants and boundary collision response, so the two can
// be diffed particle for particle -- which is what FluidSolverGpuTests does. Where the two
// necessarily differ it is because the GPU solve is order-independent and the CPU one is not; see
// the note on the Jacobi split in computePositionCorrections().
//
// Nothing is read back per frame. The particle buffer this writes is the same std430 layout the
// renderer consumes, so positions go straight from the solve to the billboard and screen-space
// liquid passes without a round trip through the host -- ResourceRegistry::readBuffer() waits for
// the device to go idle, so a per-frame readback would stall the pipeline every frame and give
// back the parallelism this exists to exploit. readParticles() is for tests and debugging only.
class FluidSolverGpu
{
public:
    // Mirrors FluidSolverCpu::Config. Held separately rather than reused directly because the GPU
    // solver needs its buffers sized up front, which a CPU solver that can resize freely does not.
    struct Config
    {
        // Simulation parameters. These have the same meaning as in FluidSolverCpu::Config and are
        // packed into a uniform buffer verbatim; see that header for what each one does.
        float     smoothingRadius = 0.1f;
        float     restDensity     = 1000.0f;
        float     particleMass    = 1.0f;
        glm::vec3 gravity{0.0f, -9.81f, 0.0f};
        float     relaxation       = 1000.0f;
        int       solverIterations = 4;
        // Ceiling on solverIterations. The solve is unrolled into declared passes, so this fixes
        // how many are built; solverIterations can then be tuned at runtime up to this without
        // rebuilding the graph. Raising it past what is used costs the skipped passes' descriptor
        // binds and barriers, so it is a budget rather than a value to set generously.
        int       maxSolverIterations = 8;
        glm::vec3 boundsMin{-0.5f, 0.0f, -0.5f};
        glm::vec3 boundsMax{0.5f, 1.0f, 0.5f};
        float     boundsRestitution        = 0.0f;
        float     boundsFriction           = 0.98f;
        float     viscosity                = 0.05f;
        float     maxSpeedInRadiiPerSecond = 20.0f;

        // Buffer capacity. Unlike the CPU solver this is fixed at construction: every buffer below
        // is allocated for it, and setParticles() rejects more than this many.
        uint32_t maxParticles = 65536;

        // Fixed simulation timestep, independent of frame rate, and the ceiling on steps per frame
        // so a slow frame cannot spiral into an ever-growing backlog -- both as in FluidDemo. The
        // ceiling also fixes how many pass chains build() declares, so it is read at build time and
        // changing it afterwards has no effect.
        float fixedDeltaTime   = 1.0f / 60.0f;
        int   maxStepsPerFrame = 4;
        bool  startPaused      = false;

        // Buckets in the neighbour hash. Rounded up to a power of two so the shader wraps with a
        // mask. Zero derives it from maxParticles at the same two-buckets-per-particle load factor
        // the CPU solver uses; unlike there it cannot be resized per step, because the buffer is
        // allocated once.
        uint32_t hashBuckets = 0;

        // Name the solved particles are published under, in ParticleGpuData layout, for the
        // renderer to bind. Defaults to the name ParticleBillboardPass already expects.
        std::string particleBufferResourceName = "fluid_particles";

        // Appearance of the published records. The solve ignores these; they are written into the
        // particle buffer so the renderer has everything it needs from the one binding.
        float     particleRadius = 0.045f;
        glm::vec3 particleColor{0.25f, 0.5f, 0.95f};
    };

    FluidSolverGpu(Config config, ResourceRegistry &resources);

    // Declares the solve. Call once, before the frame loop; step counts and parameters are
    // uniforms and push constants, so they can change per frame without rebuilding.
    void build(FrameGraph &frameGraph);

    // Uploads initial positions and zeroes their velocities. Safe to call again to reset.
    void setParticles(std::span<const glm::vec3> positions);

    // Advances by whole fixed steps covering frameDeltaTime, matching FluidDemo's fixed-step
    // accumulator. The work is recorded when the graph executes, not here.
    void update(float frameDeltaTime);
    void stepOnce();
    void setPaused(bool paused) { m_paused = paused; }
    bool paused() const { return m_paused; }

    // Re-packs the config into its uniform buffer. Call after mutating config().
    void updateParams();

    uint32_t particleCount() const { return m_particleCount; }
    // Steps the graph will run when it next executes. Zero when paused or caught up.
    uint32_t pendingSteps() const { return m_pendingSteps; }

    Config       &config() { return m_config; }
    const Config &config() const { return m_config; }

    // Positions as of the last submitted frame. Waits for the device to go idle, so this is for
    // tests and debugging, not the frame loop -- the renderer binds the buffer directly instead.
    std::vector<glm::vec3> readPositions();
    std::vector<glm::vec3> readVelocities();
    // Per-particle density from the last solver iteration, the same diagnostic FluidSolverCpu
    // exposes. Mean over rest density is the headline convergence number.
    std::vector<float> readDensities();

    // Fixed simulation timestep, independent of frame rate. Takes effect on the next update().
    void  setFixedDeltaTime(float seconds);
    float fixedDeltaTime() const { return m_fixedDeltaTime; }

    // A GPU solve of this config matched against the CPU reference needs the same parameters on
    // both sides; this builds them so a test cannot let the two drift apart.
    static FluidSolverCpu::Config toCpuConfig(const Config &config);

    // Workgroup size the shaders declare. Dispatch sizes round up to it.
    static constexpr uint32_t kWorkgroupSize = 256;
    static uint32_t           dispatchGroups(uint32_t items) { return (items + kWorkgroupSize - 1) / kWorkgroupSize; }
    // hashBuckets as the shader will actually use it: rounded up to a power of two.
    static uint32_t resolvedHashBuckets(const Config &config);

private:
    // Declares one compute pass with the full solver buffer set bound, leaving the caller to attach
    // its dispatch. The solve is a sequence of these.
    PassBuilder addComputePass(FrameGraph &frameGraph, const std::string &name, const std::string &shader);
    // The four grid passes for one step: clear, count, scan, scatter.
    void addGridPasses(FrameGraph &frameGraph, uint32_t step);
    // Whether this step's stages dispatch this frame, and whether one of its constraint iterations
    // is among the configured ones. Separate because only the density/correct/apply triple is per
    // iteration; integration and the velocity update run once per step whatever that count is.
    bool shouldDispatchStep(uint32_t step) const;
    bool shouldDispatchIteration(uint32_t step, uint32_t iteration) const;

    // Names of the device-local buffers backing the solve. Registered in the constructor.
    std::string m_positions;
    std::string m_velocities;
    std::string m_predicted;
    std::string m_corrections;
    std::string m_densities;
    std::string m_lambdas;
    std::string m_cellStarts;
    std::string m_cellCounts;
    std::string m_cellParticles;
    std::string m_particleCells;
    std::string m_sortedPositions;
    std::string m_sortedVelocities;
    std::string m_sortedPredicted;
    std::string m_sortedParticleCells;
    std::string m_particleIds;
    std::string m_sortedParticleIds;
    std::string m_params;

    Config            m_config;
    ResourceRegistry *m_resources       = nullptr;
    uint32_t          m_particleCount   = 0;
    uint32_t          m_pendingSteps    = 0;
    bool              m_paused          = false;
    float             m_fixedDeltaTime  = 1.0f / 60.0f;
    float             m_stepAccumulator = 0.0f;
    // Step chains build() declared, which caps how many a frame can run.
    uint32_t m_declaredSteps = 0;
};

} // namespace lr
