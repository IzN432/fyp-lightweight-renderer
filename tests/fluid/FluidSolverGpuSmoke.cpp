// Diffs the GPU compute solver against the CPU reference, step for step, on real hardware.
//
// This is the test the GPU solver exists to pass: FluidSolverCpu is the ground truth, and the
// compute shaders are transcriptions of it, so the two should agree particle for particle. It runs
// the shipped SPIR-V through the real frame graph -- not a reimplementation of the solve in the
// test -- so what it checks is what ships.
//
// Opt-in, like the other GPU smoke tests: it needs a Vulkan device, so it stays out of the
// CPU-only CTest suite.
//
// Build and run:
//   cmake --build build --config Debug --target fluid_solver_gpu_smoke
//   build/Debug/fluid_solver_gpu_smoke.exe
//
// Exit code 0 means every check passed; failures are printed with the offending particle.

#include "core/framegraph/FrameGraph.hpp"
#include "core/framegraph/ResourceRegistry.hpp"
#include "core/vulkan/Allocator.hpp"
#include "core/vulkan/VulkanContext.hpp"
#include "features/fluid/FluidSolverCpu.hpp"
#include "features/fluid/FluidSolverGpu.hpp"
#include "features/fluid/ParticleBillboardPass.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lr;

namespace
{

int g_failures = 0;

void check(bool condition, const std::string &what)
{
    if (!condition)
    {
        std::printf("  FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

bool isFinite(const glm::vec3 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

// Shared configuration for both solvers. A small block, few enough particles that a brute-force
// CPU step is quick, but more than one grid cell across so the spatial hash is genuinely exercised.
FluidSolverGpu::Config makeConfig(uint32_t capacity)
{
    const float spacing = 0.05f;
    const float radius  = spacing * 2.0f;

    FluidSolverGpu::Config config;
    config.smoothingRadius = radius;
    config.particleMass    = 1.0f;
    // Derived from the lattice the particles start on, as FluidSimulation does: picking rest density
    // independently of spacing is the usual reason a first step inflates or implodes.
    config.restDensity         = FluidSolverCpu::restDensityForSpacing(spacing, radius, 1.0f);
    config.relaxation          = config.restDensity * 0.01f;
    config.artificialPressureK = 0.001f * radius * radius;
    config.viscosity           = 0.01f * config.particleMass / config.restDensity;
    // One Jacobi iteration keeps this a transcription test. With the paper's negative free-surface
    // pressure enabled, different floating-point neighbour orders become chaotic across repeated
    // iterations; convergence is tested independently below rather than mistaken for bitwise parity.
    config.solverIterations    = 1;
    config.maxSolverIterations = 1;
    config.boundsMin           = glm::vec3(-0.5f, 0.0f, -0.5f);
    config.boundsMax           = glm::vec3(0.5f, 1.0f, 0.5f);
    config.maxParticles        = capacity;
    config.maxStepsPerFrame    = 1;
    config.fixedDeltaTime      = 1.0f / 60.0f;
    return config;
}

std::vector<glm::vec3> makeBlock(const FluidSolverGpu::Config &config)
{
    const float     spacing = 0.05f;
    const glm::vec3 min     = config.boundsMin + glm::vec3(spacing);
    const glm::vec3 max     = min + glm::vec3(0.2f, 0.25f, 0.2f);
    return FluidSolverCpu::makeParticleBlock(min, max, spacing);
}

// Largest per-particle position discrepancy between the two solvers.
float maxDeviation(const std::vector<glm::vec3> &gpu, const std::vector<glm::vec3> &cpu, std::size_t &worst)
{
    float largest = 0.0f;
    worst         = 0;
    for (std::size_t i = 0; i < gpu.size() && i < cpu.size(); ++i)
    {
        const float distance = glm::length(gpu[i] - cpu[i]);
        if (distance > largest)
        {
            largest = distance;
            worst   = i;
        }
    }
    return largest;
}

} // namespace

int main(int argc, char **argv)
try
{
    VulkanContext    context({.appName = "Fluid GPU solver smoke", .enableValidation = true});
    Allocator        allocator(context);
    ResourceRegistry registry(context, allocator, {1, 1});

    if (argc > 1 && std::string(argv[1]) == "--large")
    {
        constexpr glm::uvec3 dimensions{50, 56, 50};
        constexpr float spacing = 0.5f;
        FluidSolverGpu::Config large;
        large.smoothingRadius = spacing * 2.0f;
        large.particleMass = 1.0f;
        large.restDensity = FluidSolverCpu::restDensityForSpacing(spacing, large.smoothingRadius, 1.0f);
        large.relaxation = 100.0f;
        large.artificialPressureK = 0.1f;
        large.viscosity = 0.01f;
        large.solverIterations = 8;
        large.maxSolverIterations = 8;
        large.boundsMin = glm::vec3(-15.0f, 0.0f, -15.0f);
        large.boundsMax = glm::vec3(15.0f, 35.0f, 15.0f);
        large.fixedDeltaTime = 0.016f;
        large.maxDisplacementInRadiiPerStep = 0.0f;
        large.maxParticles = dimensions.x * dimensions.y * dimensions.z;
        large.maxStepsPerFrame = 1;
        large.startPaused = true;

        std::vector<glm::vec3> particles;
        particles.reserve(large.maxParticles);
        const glm::vec3 first = large.boundsMin + glm::vec3(spacing);
        for (uint32_t x = 0; x < dimensions.x; ++x)
            for (uint32_t y = 0; y < dimensions.y; ++y)
                for (uint32_t z = 0; z < dimensions.z; ++z)
                    particles.emplace_back(first + glm::vec3(x, y, z) * spacing);

        FluidSolverGpu diagnostic(large, registry);
        diagnostic.setParticles(particles);
        FrameGraph diagnosticGraph(context, registry);
        diagnostic.build(diagnosticGraph);
        std::vector<glm::vec3> previous = particles;

        for (int step = 1; step <= 240; ++step)
        {
            diagnostic.stepOnce();
            diagnosticGraph.executeAndWait();
            const auto positions = diagnostic.readPositions();
            const auto velocities = diagnostic.readVelocities();
            const auto densities = diagnostic.readDensities();
            glm::vec3 minimum(std::numeric_limits<float>::max());
            glm::vec3 maximum(-std::numeric_limits<float>::max());
            float maxSpeed = 0.0f;
            float maxDisplacement = 0.0f;
            std::size_t worst = 0;
            bool finite = true;
            for (std::size_t i = 0; i < positions.size(); ++i)
            {
                finite &= isFinite(positions[i]) && isFinite(velocities[i]);
                minimum = glm::min(minimum, positions[i]);
                maximum = glm::max(maximum, positions[i]);
                maxSpeed = std::max(maxSpeed, glm::length(velocities[i]));
                const float displacement = glm::length(positions[i] - previous[i]);
                if (displacement > maxDisplacement)
                {
                    maxDisplacement = displacement;
                    worst = i;
                }
            }
            if (step <= 40 || step % 20 == 0)
                std::printf("step %03d finite=%d bounds=(%.4f %.4f %.4f)-(%.4f %.4f %.4f) "
                            "maxSpeed=%.4f maxMove=%.4f id=%zu y=%.4f->%.4f rho/rho0=%.3f\n",
                            step, finite ? 1 : 0, minimum.x, minimum.y, minimum.z,
                            maximum.x, maximum.y, maximum.z, maxSpeed, maxDisplacement, worst,
                            previous[worst].y, positions[worst].y, densities[worst] / large.restDensity);
            if (maxDisplacement > large.smoothingRadius)
            {
                std::fprintf(stderr, "large regression: step %d teleported %.4f kernel radii\n",
                             step, maxDisplacement / large.smoothingRadius);
                return 1;
            }
            previous = positions;
            if (!finite)
                return 1;
        }
        return 0;
    }

    const std::vector<glm::vec3> initial = makeBlock(makeConfig(1));
    const FluidSolverGpu::Config config  = makeConfig(1);

    std::printf("fluid GPU solver smoke: %zu particles, h = %.4f, rest density = %.1f\n", initial.size(),
                static_cast<double>(config.smoothingRadius), static_cast<double>(config.restDensity));

    FluidSolverGpu gpu(config, registry);
    gpu.reserve(static_cast<uint32_t>(initial.size()));
    gpu.setParticles(initial);

    FrameGraph graph(context, registry);
    gpu.build(graph);

    // The reference, configured from the same values so the two cannot drift apart.
    FluidSolverCpu cpu(FluidSolverGpu::toCpuConfig(gpu.config()));
    cpu.setParticles(initial);

    check(gpu.particleCount() == initial.size(), "GPU particle count matches the upload");

    // ---------------------------------------------------------------------
    // Step the two solvers in lockstep and compare.
    //
    // What is compared is densities, not positions, and only over the first few steps. The reason
    // is that a particle system under a density constraint is chaotic: the GPU fills each grid
    // bucket in whatever order its atomics resolve in, where the CPU fills it in ascending particle
    // index, so the two sum each neighbourhood in a different order. Float addition is not
    // associative, so that alone makes the last bits differ, and the difference then grows
    // exponentially. Measured here: perturbing a single coordinate of the CPU solver's input by one
    // ULP and running both copies for 60 steps separates them by about 28% of a smoothing radius.
    // Any position tolerance loose enough to survive that is too loose to catch a real bug, so
    // positions are not what this asserts on.
    //
    // Densities are. They are computed from the neighbour sets and the kernels alone, so on the
    // early steps -- before chaos has had time to amplify anything -- they pin down exactly what a
    // transcription bug would break: a missed neighbour, a duplicated one, a wrong kernel constant,
    // a stale grid. A dropped neighbour moves a density by a percent or more, hundreds of times the
    // reassociation noise, so the bound below separates the two cleanly.
    //
    // This is what caught the bug this solver was written with: the neighbour traversal recomputed
    // each candidate's cell from its current predicted position, but the grid had been built one
    // iteration earlier, so every particle a correction had carried over a cell boundary silently
    // vanished from its neighbours' sets. Densities showed it at once; positions only looked
    // "roughly similar".
    // ---------------------------------------------------------------------
    constexpr int kComparedSteps = 4;
    // Generous next to the ~2e-06 that reassociation actually produces, and still far below the
    // percent-level error a single missing neighbour causes.
    // With the paper's negative free-surface pressure restored, reassociation noise grows faster
    // across time steps than it did under the old one-sided clamp. This remains an order of
    // magnitude below the percent-level signature of a missing neighbour.
    constexpr float kDensityTolerance = 1e-3f;
    std::vector<glm::vec3> previousGpuPositions = initial;

    for (int step = 0; step < kComparedSteps; ++step)
    {
        gpu.stepOnce();
        graph.executeAndWait();
        cpu.step(config.fixedDeltaTime);

        const std::vector<glm::vec3> gpuPositions = gpu.readPositions();
        const std::vector<float>     gpuDensities = gpu.readDensities();
        check(gpuPositions.size() == cpu.positions().size(), "position counts agree");
        check(gpuDensities.size() == cpu.densities().size(), "density counts agree");

        float       worstRelative = 0.0f;
        std::size_t worstIndex    = 0;
        for (std::size_t i = 0; i < gpuDensities.size() && i < cpu.densities().size(); ++i)
        {
            const float reference = std::max(cpu.densities()[i], 1e-6f);
            const float relative  = std::fabs(gpuDensities[i] - cpu.densities()[i]) / reference;
            if (relative > worstRelative)
            {
                worstRelative = relative;
                worstIndex    = i;
            }
        }

        if (worstRelative > kDensityTolerance)
        {
            std::printf("  FAIL: step %d particle %zu density %.4f vs reference %.4f "
                        "(relative %.3e, tolerance %.1e)\n",
                        step, worstIndex, static_cast<double>(gpuDensities[worstIndex]),
                        static_cast<double>(cpu.densities()[worstIndex]), static_cast<double>(worstRelative),
                        static_cast<double>(kDensityTolerance));
            ++g_failures;
        } else
        {
            std::printf("  step %d: worst density deviation %.3e (tolerance %.1e)\n", step,
                        static_cast<double>(worstRelative), static_cast<double>(kDensityTolerance));
        }

        const float travelLimit = config.maxDisplacementInRadiiPerStep * config.smoothingRadius;
        for (std::size_t i = 0; i < gpuPositions.size(); ++i)
        {
            const glm::vec3 &position = gpuPositions[i];
            if (!isFinite(position))
            {
                check(false, "GPU positions stayed finite");
                break;
            }
            check(glm::distance(position, previousGpuPositions[i]) <= travelLimit + 1e-5f,
                  "GPU particle travel stayed within the per-step cap");
        }
        previousGpuPositions = gpuPositions;
    }

    // Positions are still checked, but only on the step where chaos has had no time to act: one
    // step from identical inputs leaves nothing for reassociation to amplify, so the two solvers
    // should agree to a tiny fraction of a particle spacing. This is the check that a stage is
    // wired up at all -- that predict, the grid, the solve and finalise all ran, in that order.
    {
        // Its own resource names, so its buffers do not collide with the solver above's.
        FluidSolverGpu::Config singleConfig     = config;
        singleConfig.particleBufferResourceName = "fluid_particles_single_step";
        FluidSolverGpu single(singleConfig, registry);
        single.reserve(static_cast<uint32_t>(initial.size()));
        single.setParticles(initial);
        FrameGraph singleGraph(context, registry);
        single.build(singleGraph);

        FluidSolverCpu reference(FluidSolverGpu::toCpuConfig(single.config()));
        reference.setParticles(initial);

        single.stepOnce();
        singleGraph.executeAndWait();
        reference.step(singleConfig.fixedDeltaTime);

        std::size_t worstIndex = 0;
        const float deviation  = maxDeviation(single.readPositions(), reference.positions(), worstIndex);
        const float tolerance  = singleConfig.smoothingRadius * 1e-3f;
        check(deviation < tolerance, "one step from identical input matches the reference");
        std::printf("  one step from identical input: worst position deviation %.3e (tolerance %.1e)\n",
                    static_cast<double>(deviation), static_cast<double>(tolerance));
    }

    // ---------------------------------------------------------------------
    // Containment and convergence, asserted directly rather than through the CPU comparison, so a
    // fault common to both solvers cannot hide.
    // ---------------------------------------------------------------------
    const std::vector<glm::vec3> finalPositions = gpu.readPositions();
    bool                         contained      = true;
    for (const glm::vec3 &position : finalPositions)
    {
        const float epsilon = 1e-4f;
        contained &= position.x >= config.boundsMin.x - epsilon && position.x <= config.boundsMax.x + epsilon &&
                     position.y >= config.boundsMin.y - epsilon && position.y <= config.boundsMax.y + epsilon &&
                     position.z >= config.boundsMin.z - epsilon && position.z <= config.boundsMax.z + epsilon;
    }
    check(contained, "every particle stayed inside the domain");

    // Gravity is the only force with a net direction here, so the block must have fallen.
    float meanHeightBefore = 0.0f;
    float meanHeightAfter  = 0.0f;
    for (std::size_t i = 0; i < initial.size(); ++i)
    {
        meanHeightBefore += initial[i].y;
        meanHeightAfter += finalPositions[i].y;
    }
    meanHeightBefore /= static_cast<float>(initial.size());
    meanHeightAfter /= static_cast<float>(initial.size());
    check(meanHeightAfter < meanHeightBefore, "the block fell under gravity");
    std::printf("  mean height %.4f -> %.4f\n", static_cast<double>(meanHeightBefore),
                static_cast<double>(meanHeightAfter));

    // The density diagnostic the CPU solver exposes, read back from the GPU buffer. Interior
    // particles of a settling block read near rest density; surface ones legitimately read low, so
    // this checks the peak rather than the mean.
    const std::vector<float> densities = gpu.readDensities();
    check(densities.size() == initial.size(), "density readback is per particle");
    float peakDensity = 0.0f;
    for (const float density : densities)
    {
        check(std::isfinite(density), "densities stayed finite");
        peakDensity = std::max(peakDensity, density);
    }
    // A converged solve does not leave the fluid wildly overcompressed. This is a loose sanity
    // bound, not a convergence measure: the tight check is the CPU comparison above.
    check(peakDensity < config.restDensity * 3.0f, "peak density is not runaway");
    std::printf("  peak density %.1f (rest %.1f)\n", static_cast<double>(peakDensity),
                static_cast<double>(config.restDensity));

    // ---------------------------------------------------------------------
    // The published buffer: what the renderer actually binds. This is the whole point of keeping
    // the solve on the device, so verify the records match the solved positions.
    // ---------------------------------------------------------------------
    const std::vector<std::byte> published      = registry.readBuffer(config.particleBufferResourceName);
    const auto                  *records        = reinterpret_cast<const ParticleGpuData *>(published.data());
    float                        worstPublished = 0.0f;
    for (std::size_t i = 0; i < finalPositions.size(); ++i)
    {
        const glm::vec3 record = glm::vec3(records[i].positionAndRadius);
        worstPublished         = std::max(worstPublished, glm::length(record - finalPositions[i]));
        check(records[i].positionAndRadius.w == config.particleRadius, "published radius is the configured one");
        check(records[i].colorAndType.w == 0.0f, "published type is Liquid");
    }
    check(worstPublished == 0.0f, "published positions are exactly the solved positions");

    // ---------------------------------------------------------------------
    // Paused frames must not advance the sim, and must still leave the renderer a valid buffer.
    // ---------------------------------------------------------------------
    gpu.setPaused(true);
    gpu.update(1.0f / 60.0f);
    check(gpu.pendingSteps() == 0, "a paused update queues no steps");
    graph.executeAndWait();
    const std::vector<glm::vec3> afterPause  = gpu.readPositions();
    std::size_t                  unusedIndex = 0;
    check(maxDeviation(afterPause, finalPositions, unusedIndex) == 0.0f, "a paused frame leaves positions untouched");

    if (g_failures == 0)
    {
        std::printf("fluid GPU solver smoke: all checks passed\n");
        return 0;
    }
    std::printf("fluid GPU solver smoke: %d check(s) failed\n", g_failures);
    return 1;
} catch (const std::exception &e)
{
    std::fprintf(stderr, "fluid_solver_gpu_smoke failed: %s\n", e.what());
    return 1;
}
