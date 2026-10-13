#include "core/framegraph/ResourceRegistry.hpp"
#include "core/vulkan/Allocator.hpp"
#include "core/vulkan/VulkanContext.hpp"
#include "features/eulerian_particle/CudaVolume.hpp"
#include "features/eulerian_particle/CudaVolumeKernels.cuh"
#include "features/eulerian_particle/SmokeSolver.hpp"

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

int failures = 0;

void check(bool condition, const char *what)
{
    if (condition)
    {
        std::cout << "  ok: " << what << '\n';
        return;
    }
    std::cerr << "  FAILED: " << what << '\n';
    ++failures;
}

double sum(const std::vector<float> &values)
{
    double total = 0.0;
    for (float value : values)
        total += value;
    return total;
}

double maxAbs(const std::vector<float> &values)
{
    double largest = 0.0;
    for (float value : values)
        largest = std::max(largest, static_cast<double>(std::fabs(value)));
    return largest;
}

bool allFinite(const std::vector<float> &values)
{
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

// Largest density found strictly outside an origin-anchored sub-region of the allocated volume.
// A solver told to simulate only that sub-region must never write past it.
double maxDensityOutside(const std::vector<float> &density, VkExtent3D allocated, VkExtent3D active)
{
    double largest = 0.0;
    for (uint32_t z = 0; z < allocated.depth; ++z)
        for (uint32_t y = 0; y < allocated.height; ++y)
            for (uint32_t x = 0; x < allocated.width; ++x)
            {
                if (x < active.width && y < active.height && z < active.depth)
                    continue;
                const std::size_t index =
                    (static_cast<std::size_t>(z) * allocated.height + y) * allocated.width + x;
                largest = std::max(largest, static_cast<double>(std::fabs(density[index])));
            }
    return largest;
}

// Divergence of the solver's published velocity field, recomputed on the host. The projection
// stage should leave this far smaller than it was before the pressure solve ran.
double interiorDivergence(const std::vector<float> &velocity, VkExtent3D extent)
{
    const auto at = [&](int x, int y, int z, int component) {
        const std::size_t index = ((static_cast<std::size_t>(z) * extent.height + static_cast<std::size_t>(y)) *
                                       extent.width +
                                   static_cast<std::size_t>(x)) *
                                  4;
        return static_cast<double>(velocity[index + static_cast<std::size_t>(component)]);
    };

    double largest = 0.0;
    for (int z = 1; z + 1 < static_cast<int>(extent.depth); ++z)
        for (int y = 1; y + 1 < static_cast<int>(extent.height); ++y)
            for (int x = 1; x + 1 < static_cast<int>(extent.width); ++x)
            {
                const double d = 0.5 * ((at(x + 1, y, z, 0) - at(x - 1, y, z, 0)) +
                                        (at(x, y + 1, z, 1) - at(x, y - 1, z, 1)) +
                                        (at(x, y, z + 1, 2) - at(x, y, z - 1, 2)));
                largest = std::max(largest, std::fabs(d));
            }
    return largest;
}

// Largest density anywhere in an inclusive-min/exclusive-max cell range, and the total above a
// given height. Together they say whether an obstacle both stayed empty and actually blocked the
// plume rather than merely deflecting it around the edges.
double maxDensityInBox(const std::vector<float> &density, VkExtent3D extent, int3 boxMin, int3 boxMax)
{
    double largest = 0.0;
    for (int z = boxMin.z; z < boxMax.z; ++z)
        for (int y = boxMin.y; y < boxMax.y; ++y)
            for (int x = boxMin.x; x < boxMax.x; ++x)
            {
                const std::size_t index =
                    (static_cast<std::size_t>(z) * extent.height + static_cast<std::size_t>(y)) * extent.width +
                    static_cast<std::size_t>(x);
                largest = std::max(largest, static_cast<double>(std::fabs(density[index])));
            }
    return largest;
}

// Total density in a half-open cell box, for comparing how much smoke ends up in one region
// against another — the column above an obstacle versus the space beside it, say.
double densityInBox(const std::vector<float> &density, VkExtent3D extent, int3 boxMin, int3 boxMax)
{
    double total = 0.0;
    for (int z = boxMin.z; z < boxMax.z; ++z)
        for (int y = boxMin.y; y < boxMax.y; ++y)
            for (int x = boxMin.x; x < boxMax.x; ++x)
            {
                const std::size_t index =
                    (static_cast<std::size_t>(z) * extent.height + static_cast<std::size_t>(y)) * extent.width +
                    static_cast<std::size_t>(x);
                total += static_cast<double>(density[index]);
            }
    return total;
}

double densityAbove(const std::vector<float> &density, VkExtent3D extent, uint32_t minY)
{
    double total = 0.0;
    for (uint32_t z = 0; z < extent.depth; ++z)
        for (uint32_t y = minY; y < extent.height; ++y)
            for (uint32_t x = 0; x < extent.width; ++x)
                total += static_cast<double>(
                    density[(static_cast<std::size_t>(z) * extent.height + y) * extent.width + x]);
    return total;
}

} // namespace

int main()
{
    try
    {
        std::vector<const char *> externalMemoryExtensions = {VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME};
        externalMemoryExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#if defined(_WIN32)
        externalMemoryExtensions.push_back("VK_KHR_external_memory_win32");
        externalMemoryExtensions.push_back("VK_KHR_external_semaphore_win32");
#elif defined(__linux__)
        externalMemoryExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        externalMemoryExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif
        lr::VulkanContext context(lr::VulkanContext::Config{
            .appName               = "Smoke solver smoke",
            .enableValidation      = true,
            .extraDeviceExtensions = std::move(externalMemoryExtensions),
        });
        lr::Allocator        allocator(context);
        lr::ResourceRegistry resources(context, allocator, {1, 1});

        lr::eulerian::SmokeSolver::Config config{};
        config.extent           = VkExtent3D{32, 48, 32};
        config.jacobiIterations = 40;
        config.step.buoyancy    = 8.0f;
        config.step.vorticity   = 2.0f;
        config.step.densityDamp = 0.1f;
        config.source.density   = 20.0f;
        config.source.radius    = 0.08f;
        config.source.velocity[1] = 6.0f;

        lr::eulerian::SmokeSolver solver(resources, context, config);

        std::cout << "a freshly reset solver starts empty\n";
        const std::vector<float> initialDensity = solver.densityVolume().download();
        check(initialDensity.size() == 32u * 48u * 32u, "the density volume downloads one float per cell");
        check(maxAbs(initialDensity) == 0.0, "reset() zeroes the density field");
        check(maxAbs(solver.velocityVolume().download()) == 0.0, "reset() zeroes the velocity field");
        check(maxAbs(solver.pressureVolume().download()) == 0.0, "reset() zeroes the pressure field");

        std::cout << "one step injects smoke and keeps every field finite\n";
        solver.step();
        check(solver.stepPending(), "step() leaves the handshake open rather than blocking");
        solver.waitForStep();
        check(!solver.stepPending(), "waitForStep() closes the handshake");
        const std::vector<float> density1  = solver.densityVolume().download();
        const std::vector<float> velocity1 = solver.velocityVolume().download();
        check(allFinite(density1) && allFinite(velocity1), "density and velocity stay finite");
        check(sum(density1) > 0.0, "the source injects density");
        check(maxAbs(velocity1) > 0.0, "the source and buoyancy produce velocity");
        check(*std::min_element(density1.begin(), density1.end()) >= 0.0f, "density never goes negative");

        std::cout << "the projection leaves the published velocity nearly divergence-free\n";
        const double residual = interiorDivergence(velocity1, config.extent);
        check(residual < 0.05 * std::max(1.0, maxAbs(velocity1)),
              "interior divergence is small relative to the velocity magnitude");

        std::cout << "smoke rises: buoyancy moves the density centroid upward\n";
        const auto centroidY = [&](const std::vector<float> &field) {
            double weighted = 0.0;
            double total    = 0.0;
            for (uint32_t z = 0; z < config.extent.depth; ++z)
                for (uint32_t y = 0; y < config.extent.height; ++y)
                    for (uint32_t x = 0; x < config.extent.width; ++x)
                    {
                        const double value = field[(static_cast<std::size_t>(z) * config.extent.height + y) *
                                                       config.extent.width +
                                                   x];
                        weighted += value * static_cast<double>(y);
                        total += value;
                    }
            return total > 0.0 ? weighted / total : 0.0;
        };

        const double startHeight = centroidY(density1);
        // The per-frame pattern: step() hands the volumes to CUDA and endStep() takes them back,
        // with no host wait in between. Sixty of these in a row also check that the binary
        // semaphores stay balanced over a long run rather than drifting into a deadlock.
        for (int frame = 0; frame < 60; ++frame)
        {
            solver.step();
            solver.endStep();
        }
        check(!solver.stepPending(), "60 step()/endStep() pairs leave no step outstanding");
        const std::vector<float> density60 = solver.densityVolume().download();
        check(allFinite(density60), "density stays finite over 60 steps");
        check(allFinite(solver.velocityVolume().download()), "velocity stays finite over 60 steps");
        check(allFinite(solver.pressureVolume().download()), "pressure stays finite over 60 steps");
        check(centroidY(density60) > startHeight, "the density centroid rises");

        std::cout << "an explicit timestep does not disturb the configured one\n";
        solver.step(1.0f / 120.0f);
        check(solver.config().step.dt == config.step.dt, "step(dt) restores Config::step.dt");

        std::cout << "stepping twice without closing the handshake is refused\n";
        bool refused = false;
        try
        {
            solver.step();
        } catch (const std::logic_error &)
        {
            refused = true;
        }
        check(refused, "step() throws while a step is still in flight");
        solver.endStep();

        std::cout << "reset clears the simulation again\n";
        solver.reset();
        check(maxAbs(solver.densityVolume().download()) == 0.0, "reset() empties a running simulation");

        std::cout << "a solver simulating a sub-region leaves the rest of its allocation alone\n";
        {
            // The whole point of allocating at the maximum: the grid can be resized afterwards and
            // the kernels, which bound-check against the extent they are handed, simply stop short.
            lr::eulerian::SmokeSolver::Config sub = config;
            sub.namePrefix                        = "subregion";
            sub.activeExtent                      = VkExtent3D{16, 24, 16};
            lr::eulerian::SmokeSolver subSolver(resources, context, sub);
            check(subSolver.activeExtent().width == 16 && subSolver.activeExtent().height == 24 &&
                      subSolver.activeExtent().depth == 16,
                  "the active extent is the one that was configured");

            for (int i = 0; i < 30; ++i)
            {
                subSolver.step(1.0f / 60.0f);
                subSolver.endStep();
            }
            subSolver.waitForStep();

            const std::vector<float> density = subSolver.densityVolume().download();
            check(allFinite(density), "a sub-region simulation stays finite");
            check(sum(density) > 0.0, "the sub-region actually accumulated density");
            check(maxDensityOutside(density, config.extent, sub.activeExtent) == 0.0,
                  "no density is written outside the active sub-region");

            // Growing the active extent must clear: density laid down at the old size describes a
            // grid that no longer exists, so reset() is what makes the new size meaningful.
            subSolver.setActiveExtent(VkExtent3D{24, 32, 24});
            check(subSolver.activeExtent().height == 32, "setActiveExtent() retargets the solver");
            subSolver.reset();
            check(maxAbs(subSolver.densityVolume().download()) == 0.0,
                  "reset() after a resize clears the whole allocation");

            bool rejected = false;
            try
            {
                subSolver.setActiveExtent(VkExtent3D{64, 64, 64});
            } catch (const std::invalid_argument &)
            {
                rejected = true;
            }
            check(rejected, "an active extent larger than the allocation is refused");
        }

        std::cout << "an obstacle mask blocks the plume\n";
        {
            // A lid across the whole box a third of the way up, with the emitter underneath it.
            // Spanning x and z fully means there is no way around it: smoke found above the lid, or
            // inside it, can only have leaked through, which is exactly what the no-flux condition
            // in the divergence and projection stages exists to prevent.
            lr::eulerian::SmokeSolver::Config blocked = config;
            blocked.namePrefix                        = "blocked";
            blocked.obstacleMaskName                  = "blocked_obstacle_mask";
            // The solver imports the mask but never registers it: that is the voxelizer's job in
            // the application, and the test's here.
            resources.registerPersistentVolume(blocked.obstacleMaskName, VK_FORMAT_R32_UINT,
                                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                               blocked.extent, true);

            lr::eulerian::SmokeSolver blockedSolver(resources, context, blocked);
            const lr::eulerian::CudaVolume *mask = blockedSolver.obstacleVolume();
            check(mask != nullptr, "naming a mask imports it into the solver");

            const int3 slabMin{0, 16, 0};
            const int3 slabMax{static_cast<int>(blocked.extent.width), 20,
                               static_cast<int>(blocked.extent.depth)};
            if (mask != nullptr)
            {
                // Written on the null stream and synchronized, so the mask is in place before the
                // first step reads it. The solver's own handshake orders the per-frame case.
                if (cudaSetDevice(mask->device()) != cudaSuccess)
                    throw std::runtime_error("cudaSetDevice failed");
                if (lr::eulerian::launchFillObstacleBox(mask->surface(), mask->extent(), slabMin, slabMax,
                                                        nullptr) != cudaSuccess)
                    throw std::runtime_error("launchFillObstacleBox failed");
                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw std::runtime_error("cudaDeviceSynchronize failed");

                const std::vector<uint32_t> readBack = mask->downloadUint();
                const std::size_t           lidCell =
                    (static_cast<std::size_t>(0) * blocked.extent.height + 17) * blocked.extent.width + 5;
                check(readBack[lidCell] == 1u, "the mask reads back as set inside the slab");
                check(readBack[0] == 0u, "and clear outside it");
            }

            // Long enough that an unblocked plume would be well past the lid: the unobstructed run
            // above raised its centroid within 60 steps at this timestep.
            for (int frame = 0; frame < 120; ++frame)
            {
                blockedSolver.step(1.0f / 60.0f);
                blockedSolver.endStep();
            }
            blockedSolver.waitForStep();

            const std::vector<float> density = blockedSolver.densityVolume().download();
            check(allFinite(density), "the field stays finite with an obstacle in it");
            check(sum(density) > 0.0, "the source below the lid still injects density");
            check(maxDensityInBox(density, blocked.extent, slabMin, slabMax) == 0.0,
                  "no density is left inside the obstacle");

            // Some leakage through a one-cell-thick region of a collocated grid is expected; what
            // matters is that the lid holds back the great majority of the plume rather than
            // letting it stream through.
            const double above = densityAbove(density, blocked.extent, 20);
            check(above < 0.02 * sum(density), "almost no density gets past the lid");

            // The control: the same solver with the mask cleared must let the plume straight
            // through, which is what proves the checks above are measuring the obstacle and not
            // some other limit on how far the smoke rises.
            if (mask != nullptr)
            {
                blockedSolver.reset();
                if (lr::eulerian::launchFillObstacleBox(mask->surface(), mask->extent(), int3{0, 0, 0},
                                                        int3{0, 0, 0}, nullptr) != cudaSuccess)
                    throw std::runtime_error("launchFillObstacleBox failed");
                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw std::runtime_error("cudaDeviceSynchronize failed");

                for (int frame = 0; frame < 120; ++frame)
                {
                    blockedSolver.step(1.0f / 60.0f);
                    blockedSolver.endStep();
                }
                blockedSolver.waitForStep();

                const std::vector<float> freeDensity = blockedSolver.densityVolume().download();
                check(densityAbove(freeDensity, blocked.extent, 20) > 0.1 * sum(freeDensity),
                      "with the mask cleared the plume does rise past that height");
            }
        }

        std::cout << "a partial obstacle deflects the plume around itself\n";
        {
            // The scenario the full-span lid cannot test: a plate covering the middle of the box
            // with clear fluid on all four sides, so the deflected smoke has somewhere to go.
            //
            // What to expect here is worth stating, because the intuition that smoke should
            // "mostly go back up" does not hold in a sealed box. Spreading sideways on impact is
            // the correct response to u.n = 0 — the normal momentum has to become tangential — and
            // in a box whose every boundary is no-flux, the smoke pushed aside has nowhere to go
            // but down: the plume drives a recirculation cell, and the side channels carry a
            // downdraft even with no obstacle present at all. So the checks below pin the
            // deflection itself (the plate holds, the plume is pushed into the channels beside it,
            // and the obstructed run reaches the channels far more strongly than the free one),
            // not a recovery that a closed domain does not produce.
            lr::eulerian::SmokeSolver::Config partial = config;
            partial.namePrefix                        = "partial";
            partial.obstacleMaskName                  = "partial_obstacle_mask";
            resources.registerPersistentVolume(partial.obstacleMaskName, VK_FORMAT_R32_UINT,
                                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                               partial.extent, true);

            lr::eulerian::SmokeSolver       partialSolver(resources, context, partial);
            const lr::eulerian::CudaVolume *mask = partialSolver.obstacleVolume();
            check(mask != nullptr, "naming a partial mask imports it into the solver");

            // Centred on the emitter's x/z so the plume hits it head-on, two cells thick, with
            // eight clear cells on either side for the smoke to spill through.
            const int3 plateMin{8, 16, 8};
            const int3 plateMax{24, 18, 24};
            if (mask != nullptr)
            {
                if (cudaSetDevice(mask->device()) != cudaSuccess)
                    throw std::runtime_error("cudaSetDevice failed");
                if (lr::eulerian::launchFillObstacleBox(mask->surface(), mask->extent(), plateMin, plateMax,
                                                        nullptr) != cudaSuccess)
                    throw std::runtime_error("launchFillObstacleBox failed");
                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw std::runtime_error("cudaDeviceSynchronize failed");
            }

            for (int frame = 0; frame < 180; ++frame)
            {
                partialSolver.step(1.0f / 60.0f);
                partialSolver.endStep();
            }
            partialSolver.waitForStep();

            const std::vector<float> density = partialSolver.densityVolume().download();
            check(allFinite(density), "the field stays finite around a partial obstacle");
            check(maxDensityInBox(density, partial.extent, plateMin, plateMax) == 0.0,
                  "no density is left inside the partial obstacle");

            // The two clear side channels at the plate's own height, where the deflected plume has
            // to go. This is the region the deflection fills.
            const int3 leftMin{0, 16, 0};
            const int3 leftMax{8, 20, static_cast<int>(partial.extent.depth)};
            const int3 rightMin{24, 16, 0};
            const int3 rightMax{static_cast<int>(partial.extent.width), 20,
                                static_cast<int>(partial.extent.depth)};
            const double sideways = densityInBox(density, partial.extent, leftMin, leftMax) +
                                    densityInBox(density, partial.extent, rightMin, rightMax);

            // Directly beneath the plate, where the plume stacks up on impact.
            const int3   beneathMin{8, 12, 8};
            const int3   beneathMax{24, 16, 24};
            const double beneath = densityInBox(density, partial.extent, beneathMin, beneathMax);

            check(beneath > 0.0, "the plume stacks up under the plate");
            check(sideways > 0.0, "and is deflected into the channels beside it");

            // The control: the same solver, same steps, with the mask cleared. The plume then goes
            // straight up the middle, so the side channels stay comparatively empty. Requiring the
            // obstructed run to put several times more density there is what distinguishes a real
            // deflection from smoke merely diffusing outward on its own.
            if (mask != nullptr)
            {
                partialSolver.reset();
                if (lr::eulerian::launchFillObstacleBox(mask->surface(), mask->extent(), int3{0, 0, 0},
                                                        int3{0, 0, 0}, nullptr) != cudaSuccess)
                    throw std::runtime_error("launchFillObstacleBox failed");
                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw std::runtime_error("cudaDeviceSynchronize failed");

                for (int frame = 0; frame < 180; ++frame)
                {
                    partialSolver.step(1.0f / 60.0f);
                    partialSolver.endStep();
                }
                partialSolver.waitForStep();

                const std::vector<float> freeDensity = partialSolver.densityVolume().download();
                const double             freeSideways =
                    densityInBox(freeDensity, partial.extent, leftMin, leftMax) +
                    densityInBox(freeDensity, partial.extent, rightMin, rightMax);
                check(sideways > 3.0 * freeSideways,
                      "the plate pushes far more smoke into the side channels than an open box does");

                // And the converse, so the plate is shown to block rather than merely divert: the
                // column the plate occupies carries much more smoke when nothing is in the way.
                const int3   columnMin{8, 18, 8};
                const int3   columnMax{24, 24, 24};
                check(densityInBox(freeDensity, partial.extent, columnMin, columnMax) >
                          10.0 * densityInBox(density, partial.extent, columnMin, columnMax),
                      "and the column it occupies fills up once it is gone");
            }
        }


        std::cout << "a solver destroyed mid-step tears down cleanly\n";
        {
            // Destroying a solver with an open handshake has to close it: the CUDA-side signal of
            // the return semaphore is otherwise left unconsumed while the semaphores are freed.
            // A hang or a validation error here is the failure; reaching the next line is the pass.
            lr::eulerian::SmokeSolver doomed(resources, context, [&] {
                lr::eulerian::SmokeSolver::Config other = config;
                other.namePrefix                        = "doomed";
                return other;
            }());
            doomed.step();
            check(doomed.stepPending(), "the solver under test really is mid-step");
        }
        check(true, "destroying a mid-step solver completes without hanging");

        if (failures != 0)
        {
            std::cerr << failures << " check(s) failed\n";
            return 1;
        }
        std::cout << "all smoke solver checks passed\n";
        return 0;
    } catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
