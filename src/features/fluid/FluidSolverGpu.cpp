#include "features/fluid/FluidSolverGpu.hpp"

#include "core/Paths.hpp"
#include "features/fluid/ParticleBillboardPass.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace lr
{
namespace
{

// Mirrors the FluidParams block in fluid_sim.glslh. std140: every member is a four-component vector
// so the layout is unambiguous on both sides, and vec3s are padded explicitly rather than relying
// on the rule that pads them -- a bare vec3 followed by a float is the classic way for a uniform
// block to silently disagree with its shader.
struct alignas(16) GpuFluidParams
{
    glm::vec4  gravityAndDeltaTime;
    glm::vec4  boundsMin;
    glm::vec4  boundsMax;
    glm::vec4  massRelaxViscosity;
    glm::vec4  restitutionFrictionSpeed;
    glm::vec4  pressureAndVorticity;
    glm::uvec4 counts;
};

static_assert(sizeof(GpuFluidParams) == 112);

// The appearance push constant in fluid_publish.comp.
struct alignas(16) GpuAppearance
{
    glm::vec4 colorAndRadius;
};

static_assert(sizeof(GpuAppearance) == 16);

uint32_t roundUpToPowerOfTwo(uint32_t value)
{
    uint32_t result = 1;
    while (result < value)
    {
        result <<= 1;
    }
    return result;
}

} // namespace

uint32_t FluidSolverGpu::resolvedHashBuckets(const Config &config)
{
    // Two buckets per particle keeps the load factor low enough that a bucket holds about one
    // cell's worth of particles, which is the point of the grid. Unlike the CPU solver this is
    // sized from the capacity rather than the live count, because the buffer is allocated once.
    const uint32_t requested = config.hashBuckets != 0 ? config.hashBuckets : config.maxParticles * 2;
    return roundUpToPowerOfTwo(std::max(requested, 1u));
}

FluidSolverCpu::Config FluidSolverGpu::toCpuConfig(const Config &config)
{
    FluidSolverCpu::Config cpu;
    cpu.smoothingRadius          = config.smoothingRadius;
    cpu.restDensity              = config.restDensity;
    cpu.particleMass             = config.particleMass;
    cpu.gravity                  = config.gravity;
    cpu.relaxation               = config.relaxation;
    cpu.artificialPressureK        = config.artificialPressureK;
    cpu.artificialPressureDeltaQ   = config.artificialPressureDeltaQ;
    cpu.artificialPressureExponent = config.artificialPressureExponent;
    cpu.vorticityEpsilon           = config.vorticityEpsilon;
    cpu.solverIterations         = config.solverIterations;
    cpu.boundsMin                = config.boundsMin;
    cpu.boundsMax                = config.boundsMax;
    cpu.boundsRestitution        = config.boundsRestitution;
    cpu.boundsFriction           = config.boundsFriction;
    cpu.viscosity                = config.viscosity;
    cpu.maxSpeedInRadiiPerSecond = config.maxSpeedInRadiiPerSecond;
    // The GPU table is fixed, so pin the CPU one to the same size. Bucket counts change which
    // cells collide, and a collision changes nothing about the neighbour sets -- the distance test
    // decides membership -- but matching them keeps the two solvers comparable step for step.
    cpu.hashBuckets = resolvedHashBuckets(config);
    return cpu;
}

FluidSolverGpu::FluidSolverGpu(Config config, ResourceRegistry &resources)
    : m_config(std::move(config)), m_resources(&resources)
{
    if (m_config.maxParticles == 0)
    {
        throw std::invalid_argument("FluidSolverGpu maxParticles must be greater than zero");
    }
    if (m_config.particleBufferResourceName.empty())
    {
        throw std::invalid_argument("FluidSolverGpu requires a particle buffer resource name");
    }

    m_fixedDeltaTime = std::max(m_config.fixedDeltaTime, 1e-5f);
    m_paused         = m_config.startPaused;

    // Buffer names are derived from the published particle buffer's name so two solvers can coexist
    // in one registry without colliding.
    const std::string prefix = m_config.particleBufferResourceName;
    m_positions              = prefix + "_positions";
    m_velocities             = prefix + "_velocities";
    m_predicted              = prefix + "_predicted";
    m_corrections            = prefix + "_corrections";
    m_densities              = prefix + "_densities";
    m_lambdas                = prefix + "_lambdas";
    m_vorticities            = prefix + "_vorticities";
    m_cellStarts             = prefix + "_cell_starts";
    m_cellCounts             = prefix + "_cell_counts";
    m_cellParticles          = prefix + "_cell_particles";
    m_particleCells          = prefix + "_particle_cells";
    m_sortedPositions        = prefix + "_sorted_positions";
    m_sortedVelocities       = prefix + "_sorted_velocities";
    m_sortedPredicted        = prefix + "_sorted_predicted";
    m_sortedParticleCells    = prefix + "_sorted_particle_cells";
    m_particleIds            = prefix + "_particle_ids";
    m_sortedParticleIds      = prefix + "_sorted_particle_ids";
    m_params                 = prefix + "_params";

    const uint32_t capacity = m_config.maxParticles;
    const uint32_t buckets  = resolvedHashBuckets(m_config);

    // Positions and velocities are uploaded from the host once per reset, so they are static
    // buffers with TRANSFER_DST; everything else is pure GPU scratch. All of them are device-local:
    // a dynamic (host-visible) buffer cannot be written by a shader, which is what rules out
    // reusing the registry's per-frame buffers here.
    const VkBufferUsageFlags storage  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkBufferUsageFlags uploaded = storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // vec4 rather than vec3: a std430 array of vec3 has a 16-byte stride regardless, so storing the
    // padding explicitly keeps the C++ and GLSL indexing identical.
    const VkDeviceSize vectorBytes = VkDeviceSize(capacity) * sizeof(glm::vec4);
    const VkDeviceSize scalarBytes = VkDeviceSize(capacity) * sizeof(float);

    resources.registerStaticBuffer(m_positions, vectorBytes, uploaded);
    resources.registerStaticBuffer(m_velocities, vectorBytes, uploaded);
    resources.registerStaticBuffer(m_predicted, vectorBytes, storage);
    resources.registerStaticBuffer(m_corrections, vectorBytes, storage);
    resources.registerStaticBuffer(m_densities, scalarBytes, storage);
    resources.registerStaticBuffer(m_lambdas, scalarBytes, storage);
    resources.registerStaticBuffer(m_vorticities, vectorBytes, storage);
    // One extra entry so a bucket's span reads as [starts[b], starts[b + 1]) without the last
    // bucket needing a special case.
    resources.registerStaticBuffer(m_cellStarts, VkDeviceSize(buckets + 1) * sizeof(uint32_t), storage);
    resources.registerStaticBuffer(m_cellCounts, VkDeviceSize(buckets) * sizeof(uint32_t), storage);
    resources.registerStaticBuffer(m_cellParticles, VkDeviceSize(capacity) * sizeof(uint32_t), storage);
    // ivec4 rather than ivec3 for the same reason the positions are vec4: a std430 array of
    // three-component vectors is laid out with a 16-byte stride anyway.
    resources.registerStaticBuffer(m_particleCells, VkDeviceSize(capacity) * sizeof(glm::ivec4), storage);
    // Sebastian Lague's key locality trick: after the counting sort, reorder the hot particle
    // arrays into bucket order. The iterative neighbour passes then read contiguous particle data
    // rather than following a different indirect index for every lane. These are the temporary
    // targets; a second pass copies them back before the solve begins.
    resources.registerStaticBuffer(m_sortedPositions, vectorBytes, storage);
    resources.registerStaticBuffer(m_sortedVelocities, vectorBytes, storage);
    resources.registerStaticBuffer(m_sortedPredicted, vectorBytes, storage);
    resources.registerStaticBuffer(m_sortedParticleCells, VkDeviceSize(capacity) * sizeof(glm::ivec4), storage);
    // Reordering makes array indices unstable. Preserve a small logical ID so diagnostic readbacks
    // can still return the particle order supplied to setParticles(); rendering does not need it.
    resources.registerStaticBuffer(m_particleIds, VkDeviceSize(capacity) * sizeof(uint32_t), uploaded);
    resources.registerStaticBuffer(m_sortedParticleIds, VkDeviceSize(capacity) * sizeof(uint32_t), storage);

    // The renderer's view of the solve. Registered here rather than by the render pass so the
    // solver owns the buffer it writes; ParticleBillboardPass and ScreenSpaceLiquidPass take the
    // name through their config and bind it read-only.
    // VERTEX_BUFFER as well as STORAGE: the screen-space liquid and billboard passes read these
    // records through a storage buffer, while FluidRenderer binds the same memory as instance data,
    // reading the leading vec4 of centre and radius in place.
    resources.registerStaticBuffer(m_config.particleBufferResourceName,
                                   VkDeviceSize(capacity) * sizeof(ParticleGpuData),
                                   storage | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    resources.registerDynamicBuffer(m_params, sizeof(GpuFluidParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    updateParams();
}

void FluidSolverGpu::updateParams()
{
    const float h = std::max(m_config.smoothingRadius, 1e-6f);

    const GpuFluidParams params{
        .gravityAndDeltaTime = glm::vec4(m_config.gravity, std::max(m_fixedDeltaTime, 1e-5f)),
        .boundsMin           = glm::vec4(m_config.boundsMin, h),
        .boundsMax           = glm::vec4(m_config.boundsMax, std::max(m_config.restDensity, 1e-6f)),
        .massRelaxViscosity = glm::vec4(m_config.particleMass, m_config.relaxation, m_config.viscosity, 0.0f),
        .restitutionFrictionSpeed =
            glm::vec4(m_config.boundsRestitution, m_config.boundsFriction, m_config.maxSpeedInRadiiPerSecond, 0.0f),
        .pressureAndVorticity =
            glm::vec4(m_config.artificialPressureK, m_config.artificialPressureDeltaQ,
                      m_config.artificialPressureExponent, m_config.vorticityEpsilon),
        .counts = glm::uvec4(m_particleCount, resolvedHashBuckets(m_config), resolvedHashBuckets(m_config) - 1u,
                             static_cast<uint32_t>(std::max(m_config.solverIterations, 0))),
    };
    m_resources->updateBuffer(m_params, &params, sizeof(params));
}

void FluidSolverGpu::setParticles(std::span<const glm::vec3> positions)
{
    m_particleCount = particle_billboard::checkedParticleCount(positions.size(), m_config.maxParticles);

    // Padded to vec4 to match the shader's indexing, and zero-filled so the unused tail is defined
    // if anything ever reads past the live count.
    std::vector<glm::vec4> padded(m_config.maxParticles, glm::vec4(0.0f));
    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        padded[i] = glm::vec4(positions[i], 0.0f);
    }
    const VkDeviceSize bytes = VkDeviceSize(m_config.maxParticles) * sizeof(glm::vec4);
    m_resources->reuploadBuffer(m_positions, padded.data(), bytes);

    // Velocities start at zero, as in FluidSolverCpu::setParticles. The padding above is already
    // zero throughout, so it doubles as the velocity payload.
    std::fill(padded.begin(), padded.end(), glm::vec4(0.0f));
    m_resources->reuploadBuffer(m_velocities, padded.data(), bytes);

    std::vector<uint32_t> particleIds(m_config.maxParticles);
    for (uint32_t i = 0; i < m_config.maxParticles; ++i)
    {
        particleIds[i] = i;
    }
    m_resources->reuploadBuffer(m_particleIds, particleIds.data(),
                                VkDeviceSize(m_config.maxParticles) * sizeof(uint32_t));

    m_stepAccumulator = 0.0f;
    m_pendingSteps    = 0;
    updateParams();
}

void FluidSolverGpu::setFixedDeltaTime(float seconds)
{
    m_fixedDeltaTime = std::max(seconds, 1e-5f);
    // The timestep is part of the uniform the shaders read, not a push constant.
    updateParams();
}

void FluidSolverGpu::stepOnce() { m_pendingSteps = std::min(1u, m_declaredSteps); }

void FluidSolverGpu::update(float frameDeltaTime)
{
    if (m_paused || frameDeltaTime <= 0.0f)
    {
        m_pendingSteps = 0;
        return;
    }

    const float step = std::max(m_fixedDeltaTime, 1e-5f);
    m_stepAccumulator += frameDeltaTime;

    // Capped by what build() declared as well as by the configured ceiling: a step with no pass
    // chain behind it cannot run, and silently dropping it is better than pretending it did.
    const uint32_t maxSteps = m_declaredSteps;
    uint32_t       steps    = 0;
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
    m_pendingSteps = steps;
}

PassBuilder FluidSolverGpu::addComputePass(FrameGraph &frameGraph, const std::string &name, const std::string &shader)
{
    // Every stage binds the whole buffer set at the same bindings, which is what lets them share
    // fluid_sim.glslh. A binding a given shader does not touch is reported by the frame graph's
    // shader-interface check as a harmless note, not an error, so one declaration serves them all.
    return frameGraph.addPass(name)
        .type(PassType::Compute)
        .computeShader((paths::shaderDir / (shader + ".spv")).string())
        .uniformBuffer(0, frameGraph.buffer(m_params), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferReadWrite(1, frameGraph.buffer(m_positions), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferReadWrite(2, frameGraph.buffer(m_velocities), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferReadWrite(3, frameGraph.buffer(m_predicted), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferReadWrite(4, frameGraph.buffer(m_corrections), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferReadWrite(5, frameGraph.buffer(m_densities), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferReadWrite(6, frameGraph.buffer(m_lambdas), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferReadWrite(7, frameGraph.buffer(m_cellStarts), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferReadWrite(8, frameGraph.buffer(m_cellCounts), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferReadWrite(9, frameGraph.buffer(m_cellParticles), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferReadWrite(11, frameGraph.buffer(m_particleCells), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferReadWrite(18, frameGraph.buffer(m_vorticities), VK_SHADER_STAGE_COMPUTE_BIT);
}

// Whether the stages of `step` run this frame. Passes are declared once, at build time, but how
// many steps to run is known only per frame, so the surplus chains dispatch nothing. A skipped
// stage still costs its descriptor bind and barriers, which is why maxStepsPerFrame and
// maxSolverIterations bound how many are declared at all.
bool FluidSolverGpu::shouldDispatchStep(uint32_t step) const { return m_particleCount != 0 && step < m_pendingSteps; }

// Whether a constraint iteration within a running step is one of the configured ones.
//
// Kept separate from shouldDispatchStep deliberately: only the density/correct/apply triple is per
// iteration. Gating predict, finalise and viscosity on the iteration count as well would mean that
// at solverIterations = 0 the step integrated gravity but never committed the result -- which is
// exactly how the integration path is meant to be testable in isolation, as FluidSolverCpu's
// unconditional resolveBoundaryCollisions() allows.
bool FluidSolverGpu::shouldDispatchIteration(uint32_t step, uint32_t iteration) const
{
    return shouldDispatchStep(step) && iteration < static_cast<uint32_t>(std::max(m_config.solverIterations, 0));
}

void FluidSolverGpu::addGridPasses(FrameGraph &frameGraph, uint32_t step)
{
    const std::string suffix  = "_s" + std::to_string(step);
    const uint32_t    buckets = resolvedHashBuckets(m_config);

    // Dispatched over buckets rather than particles: this clears the table, so it has to cover
    // every bucket whatever the live particle count is.
    addComputePass(frameGraph, "fluid_grid_clear" + suffix, "fluid_grid_clear.comp")
        .execute([this, step, buckets](PassContext &context) {
            if (!shouldDispatchStep(step))
            {
                return;
            }
            context.cmd().dispatch(dispatchGroups(buckets), 1, 1);
        });

    addComputePass(frameGraph, "fluid_grid_count" + suffix, "fluid_grid_count.comp")
        .execute([this, step](PassContext &context) {
            if (shouldDispatchStep(step))
            {
                context.cmd().dispatch(dispatchGroups(m_particleCount), 1, 1);
            }
        });

    // Exactly one workgroup: the scan strides over the whole table itself and carries between
    // chunks in shared memory, so a second group would race it. See fluid_grid_scan.comp.
    addComputePass(frameGraph, "fluid_grid_scan" + suffix, "fluid_grid_scan.comp")
        .execute([this, step](PassContext &context) {
            if (shouldDispatchStep(step))
            {
                context.cmd().dispatch(1, 1, 1);
            }
        });

    addComputePass(frameGraph, "fluid_grid_scatter" + suffix, "fluid_grid_scatter.comp")
        .execute([this, step](PassContext &context) {
            if (shouldDispatchStep(step))
            {
                context.cmd().dispatch(dispatchGroups(m_particleCount), 1, 1);
            }
        });
}

void FluidSolverGpu::build(FrameGraph &frameGraph)
{
    // The solve is unrolled into declared passes: one chain per fixed step, and within it one
    // density/correct/apply triple per Jacobi iteration. The frame graph inserts the buffer
    // barriers between them, which is the reason for unrolling rather than looping dispatches
    // inside a single execute() -- consecutive iterations read what the previous one wrote, and
    // CommandBuffer exposes no barrier of its own.
    const uint32_t steps      = static_cast<uint32_t>(std::max(m_config.maxStepsPerFrame, 1));
    const uint32_t iterations = static_cast<uint32_t>(std::max(m_config.maxSolverIterations, 1));
    m_declaredSteps           = steps;

    // Runs whenever its step does, whatever the iteration count: integration and the velocity
    // update are once per step.
    const auto stepStage = [&](const std::string &name, const std::string &shader, uint32_t step) {
        addComputePass(frameGraph, name, shader).execute([this, step](PassContext &context) {
            if (shouldDispatchStep(step))
            {
                context.cmd().dispatch(dispatchGroups(m_particleCount), 1, 1);
            }
        });
    };

    // Runs only for the iterations the config asks for, so solverIterations can be lowered at
    // runtime without rebuilding the graph.
    const auto iterationStage = [&](const std::string &name, const std::string &shader, uint32_t step,
                                    uint32_t iteration) {
        addComputePass(frameGraph, name, shader).execute([this, step, iteration](PassContext &context) {
            if (shouldDispatchIteration(step, iteration))
            {
                context.cmd().dispatch(dispatchGroups(m_particleCount), 1, 1);
            }
        });
    };

    for (uint32_t step = 0; step < steps; ++step)
    {
        const std::string suffix = "_s" + std::to_string(step);

        stepStage("fluid_predict" + suffix, "fluid_predict.comp", step);

        // Built once per step from the predicted positions rather than per iteration: the
        // corrections are small relative to the smoothing radius, so the neighbour sets barely
        // change within a step. This matches FluidSolverCpu::step.
        addGridPasses(frameGraph, step);
        addComputePass(frameGraph, "fluid_reorder" + suffix, "fluid_reorder.comp")
            .storageBufferWrite(12, frameGraph.buffer(m_sortedPositions), VK_SHADER_STAGE_COMPUTE_BIT)
            .storageBufferWrite(13, frameGraph.buffer(m_sortedVelocities), VK_SHADER_STAGE_COMPUTE_BIT)
            .storageBufferWrite(14, frameGraph.buffer(m_sortedPredicted), VK_SHADER_STAGE_COMPUTE_BIT)
            .storageBufferWrite(15, frameGraph.buffer(m_sortedParticleCells), VK_SHADER_STAGE_COMPUTE_BIT)
            .storageBufferRead(16, frameGraph.buffer(m_particleIds), VK_SHADER_STAGE_COMPUTE_BIT)
            .storageBufferWrite(17, frameGraph.buffer(m_sortedParticleIds), VK_SHADER_STAGE_COMPUTE_BIT)
            .execute([this, step](PassContext &context) {
                if (shouldDispatchStep(step))
                {
                    context.cmd().dispatch(dispatchGroups(m_particleCount), 1, 1);
                }
            });
        addComputePass(frameGraph, "fluid_reorder_copyback" + suffix, "fluid_reorder_copyback.comp")
            .storageBufferRead(12, frameGraph.buffer(m_sortedPositions), VK_SHADER_STAGE_COMPUTE_BIT)
            .storageBufferRead(13, frameGraph.buffer(m_sortedVelocities), VK_SHADER_STAGE_COMPUTE_BIT)
            .storageBufferRead(14, frameGraph.buffer(m_sortedPredicted), VK_SHADER_STAGE_COMPUTE_BIT)
            .storageBufferRead(15, frameGraph.buffer(m_sortedParticleCells), VK_SHADER_STAGE_COMPUTE_BIT)
            .storageBufferWrite(16, frameGraph.buffer(m_particleIds), VK_SHADER_STAGE_COMPUTE_BIT)
            .storageBufferRead(17, frameGraph.buffer(m_sortedParticleIds), VK_SHADER_STAGE_COMPUTE_BIT)
            .execute([this, step](PassContext &context) {
                if (shouldDispatchStep(step))
                {
                    context.cmd().dispatch(dispatchGroups(m_particleCount), 1, 1);
                }
            });

        for (uint32_t iteration = 0; iteration < iterations; ++iteration)
        {
            const std::string it = suffix + "_i" + std::to_string(iteration);
            iterationStage("fluid_density" + it, "fluid_density.comp", step, iteration);
            iterationStage("fluid_correct" + it, "fluid_correct.comp", step, iteration);
            iterationStage("fluid_apply" + it, "fluid_apply.comp", step, iteration);
        }

        stepStage("fluid_finalise" + suffix, "fluid_finalise.comp", step);
        stepStage("fluid_vorticity" + suffix, "fluid_vorticity.comp", step);
        stepStage("fluid_vorticity_apply" + suffix, "fluid_vorticity_apply.comp", step);
        stepStage("fluid_viscosity" + suffix, "fluid_viscosity.comp", step);
        stepStage("fluid_viscosity_apply" + suffix, "fluid_viscosity_apply.comp", step);
    }

    // Packs the solved positions into the renderer's layout. Runs once per frame, after the last
    // step, and unconditionally: with the sim paused the renderer still needs a populated buffer.
    addComputePass(frameGraph, "fluid_publish", "fluid_publish.comp")
        .pushConstantSize(sizeof(GpuAppearance), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferRead(16, frameGraph.buffer(m_particleIds), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferWrite(10, frameGraph.buffer(m_config.particleBufferResourceName), VK_SHADER_STAGE_COMPUTE_BIT)
        .execute([this](PassContext &context) {
            if (m_particleCount == 0)
            {
                return;
            }
            const GpuAppearance appearance{
                .colorAndRadius = glm::vec4(m_config.particleColor, m_config.particleRadius),
            };
            context.cmd().pushConstants(context.pipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, appearance);
            context.cmd().dispatch(dispatchGroups(m_particleCount), 1, 1);
        });
}

std::vector<glm::vec3> FluidSolverGpu::readPositions()
{
    const std::vector<std::byte> bytes = m_resources->readBuffer(m_positions);
    const std::vector<std::byte> idBytes = m_resources->readBuffer(m_particleIds);
    const auto                  *data  = reinterpret_cast<const glm::vec4 *>(bytes.data());
    const auto                  *ids   = reinterpret_cast<const uint32_t *>(idBytes.data());

    std::vector<glm::vec3> positions(m_particleCount);
    for (uint32_t i = 0; i < m_particleCount; ++i)
    {
        positions[ids[i]] = glm::vec3(data[i]);
    }
    return positions;
}

std::vector<glm::vec3> FluidSolverGpu::readVelocities()
{
    const std::vector<std::byte> bytes = m_resources->readBuffer(m_velocities);
    const std::vector<std::byte> idBytes = m_resources->readBuffer(m_particleIds);
    const auto                  *data  = reinterpret_cast<const glm::vec4 *>(bytes.data());
    const auto                  *ids   = reinterpret_cast<const uint32_t *>(idBytes.data());

    std::vector<glm::vec3> velocities(m_particleCount);
    for (uint32_t i = 0; i < m_particleCount; ++i)
    {
        velocities[ids[i]] = glm::vec3(data[i]);
    }
    return velocities;
}

std::vector<float> FluidSolverGpu::readDensities()
{
    const std::vector<std::byte> bytes = m_resources->readBuffer(m_densities);
    const std::vector<std::byte> idBytes = m_resources->readBuffer(m_particleIds);
    const auto                  *data  = reinterpret_cast<const float *>(bytes.data());
    const auto                  *ids   = reinterpret_cast<const uint32_t *>(idBytes.data());

    std::vector<float> densities(m_particleCount);
    for (uint32_t i = 0; i < m_particleCount; ++i)
    {
        densities[ids[i]] = data[i];
    }
    return densities;
}

} // namespace lr
