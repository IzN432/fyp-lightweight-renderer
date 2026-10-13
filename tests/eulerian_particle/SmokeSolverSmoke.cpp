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

        std::cout << "an open top lets the plume leave instead of recirculating\n";
        {
            // The sealed box is what makes a plume fold back on itself: with every face no-flux,
            // the fluid a rising plume displaces can only return downward. openTop replaces the
            // top face's conditions with outflow — Dirichlet p = 0 in the solve, and a velocity
            // border that copies outward instead of reflecting — so the displaced fluid leaves.
            //
            // Both runs here are unobstructed, because this is a property of the domain and not of
            // the obstacle handling.
            lr::eulerian::SmokeSolver::Config sealed = config;
            sealed.namePrefix                        = "sealed";
            lr::eulerian::SmokeSolver sealedSolver(resources, context, sealed);

            lr::eulerian::SmokeSolver::Config open = config;
            open.namePrefix                        = "open";
            open.step.openTop                      = true;
            lr::eulerian::SmokeSolver openSolver(resources, context, open);

            for (int frame = 0; frame < 180; ++frame)
            {
                sealedSolver.step(1.0f / 60.0f);
                sealedSolver.endStep();
                openSolver.step(1.0f / 60.0f);
                openSolver.endStep();
            }
            sealedSolver.waitForStep();
            openSolver.waitForStep();

            const std::vector<float> sealedDensity = sealedSolver.densityVolume().download();
            const std::vector<float> openDensity   = openSolver.densityVolume().download();
            check(allFinite(sealedDensity) && allFinite(openDensity),
                  "both the sealed and the open box stay finite");

            // Smoke leaves through the open face, so the open box has to hold strictly less of it
            // after the same number of steps against the same source. This is the outflow itself:
            // nothing else in the step removes mass except densityDamp, which both runs share.
            check(sum(openDensity) < sum(sealedDensity),
                  "the open box holds less smoke than the sealed one");

            // And the mechanism, not just the mass: in the sealed box the displaced fluid turns
            // back down the sides, so the lower corners away from the plume collect smoke the
            // recirculation has dragged back. Measured in a shell of side columns below the
            // source's height, which a plume rising straight up and out cannot reach on its own.
            const int3 cornerMin{0, 2, 0};
            const int3 cornerMax{4, 8, 4};
            const double sealedCorner = densityInBox(sealedDensity, sealed.extent, cornerMin, cornerMax);
            const double openCorner   = densityInBox(openDensity, open.extent, cornerMin, cornerMax);
            check(sealedCorner > openCorner,
                  "the sealed box drags smoke back down into its corners and the open one does less");

            // The open face must vent rather than act as a lid the plume piles up against, so the
            // top row itself stays comparatively thin: mirroring density outward there (the closed
            // box's Neumann copy) is exactly what would make it accumulate.
            const int3 topMin{0, static_cast<int>(open.extent.height) - 1, 0};
            const int3 topMax{static_cast<int>(open.extent.width), static_cast<int>(open.extent.height),
                              static_cast<int>(open.extent.depth)};
            check(densityInBox(openDensity, open.extent, topMin, topMax) < 0.1 * sum(openDensity),
                  "smoke does not pile up against the open face");
        }

        std::cout << "a mask covering the open face does not seal it\n";
        {
            // Scene geometry is voxelized into the same grid the solver steps, so a ceiling lands
            // in the top row like anything else. Every kernel tests isSolid before onOpenFace, so
            // without the override in isSolid a mask there turns the vent back into a lid while the
            // toggle still reads "open" — the plume then spreads along the underside of the mask
            // instead of leaving, which is the failure this pins.
            lr::eulerian::SmokeSolver::Config ceiling = config;
            ceiling.namePrefix                        = "ceiling";
            ceiling.step.openTop                      = true;
            ceiling.obstacleMaskName                  = "ceiling_obstacle_mask";
            resources.registerPersistentVolume(ceiling.obstacleMaskName, VK_FORMAT_R32_UINT,
                                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                               ceiling.extent, true);

            lr::eulerian::SmokeSolver       ceilingSolver(resources, context, ceiling);
            const lr::eulerian::CudaVolume *mask = ceilingSolver.obstacleVolume();
            check(mask != nullptr, "naming a ceiling mask imports it into the solver");

            // Exactly the open face's own row, which is the case the override covers: geometry
            // flush with the top of the box.
            const int  top = static_cast<int>(ceiling.extent.height);
            const int3 ceilingMin{0, top - 1, 0};
            const int3 ceilingMax{static_cast<int>(ceiling.extent.width), top,
                                  static_cast<int>(ceiling.extent.depth)};
            if (mask != nullptr)
            {
                if (cudaSetDevice(mask->device()) != cudaSuccess)
                    throw std::runtime_error("cudaSetDevice failed");
                if (lr::eulerian::launchFillObstacleBox(mask->surface(), mask->extent(), ceilingMin, ceilingMax,
                                                        nullptr) != cudaSuccess)
                    throw std::runtime_error("launchFillObstacleBox failed");
                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw std::runtime_error("cudaDeviceSynchronize failed");
            }

            for (int frame = 0; frame < 180; ++frame)
            {
                ceilingSolver.step(1.0f / 60.0f);
                ceilingSolver.endStep();
            }
            ceilingSolver.waitForStep();

            const std::vector<float> ceilingDensity = ceilingSolver.densityVolume().download();
            check(allFinite(ceilingDensity), "the masked-ceiling run stays finite");

            // The vent still carries mass out, so the box holds less than the same scene with the
            // toggle off. Run here rather than reusing the sealed box above, because this one has
            // the mask in it and the comparison has to isolate the vent, not the geometry.
            lr::eulerian::SmokeSolver::Config shut = ceiling;
            shut.namePrefix                        = "ceiling_shut";
            shut.step.openTop                      = false;
            shut.obstacleMaskName                  = "ceiling_shut_obstacle_mask";
            resources.registerPersistentVolume(shut.obstacleMaskName, VK_FORMAT_R32_UINT,
                                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, shut.extent,
                                               true);
            lr::eulerian::SmokeSolver       shutSolver(resources, context, shut);
            const lr::eulerian::CudaVolume *shutMask = shutSolver.obstacleVolume();
            if (shutMask != nullptr)
            {
                if (cudaSetDevice(shutMask->device()) != cudaSuccess)
                    throw std::runtime_error("cudaSetDevice failed");
                if (lr::eulerian::launchFillObstacleBox(shutMask->surface(), shutMask->extent(), ceilingMin,
                                                        ceilingMax, nullptr) != cudaSuccess)
                    throw std::runtime_error("launchFillObstacleBox failed");
                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw std::runtime_error("cudaDeviceSynchronize failed");
            }
            for (int frame = 0; frame < 180; ++frame)
            {
                shutSolver.step(1.0f / 60.0f);
                shutSolver.endStep();
            }
            shutSolver.waitForStep();
            const std::vector<float> shutDensity = shutSolver.densityVolume().download();

            check(sum(ceilingDensity) < sum(shutDensity),
                  "a mask on the open face still lets smoke leave");

            // And the shape: a sealed ceiling makes the plume run sideways along its underside, so
            // the band just below the top fills the outer columns. Measured in the four side
            // columns of the rows beneath the face, which a venting plume leaves comparatively
            // empty.
            const auto edgeBandFraction = [&](const std::vector<float> &field) {
                double band = 0.0;
                double edge = 0.0;
                const int w = static_cast<int>(ceiling.extent.width);
                const int d = static_cast<int>(ceiling.extent.depth);
                for (int y = top - 8; y < top - 1; ++y)
                    for (int z = 0; z < d; ++z)
                        for (int x = 0; x < w; ++x)
                        {
                            const double v = field[(static_cast<std::size_t>(z) * ceiling.extent.height + y) *
                                                       ceiling.extent.width +
                                                   x];
                            band += v;
                            if (x < w / 4 || x >= 3 * w / 4 || z < d / 4 || z >= 3 * d / 4)
                                edge += v;
                        }
                return band > 1e-6 ? edge / band : 0.0;
            };
            check(edgeBandFraction(ceilingDensity) < 0.25,
                  "the plume vents through the masked face instead of spreading along it");
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

            // The same plate in an open box, which is where the deflection is supposed to recover:
            // once the displaced fluid can leave through the top, the smoke that went around the
            // edges climbs instead of being dragged back under. Measured in the column above the
            // plate, which the sealed run leaves all but empty.
            //
            // Reuses this solver rather than building a second one, both because importing another
            // full set of volumes is what exhausts the external-memory handles on some drivers and
            // because flipping the flag on a live solver is how the editor's toggle works: it is a
            // step parameter, so it takes effect on the next step with no restart.
            const int3 overheadMin{8, 20, 8};
            const int3 overheadMax{24, static_cast<int>(partial.extent.height), 24};
            const double sealedOverhead = densityInBox(density, partial.extent, overheadMin, overheadMax);
            {
                partialSolver.reset();
                partialSolver.stepParams().openTop = true;
                for (int frame = 0; frame < 180; ++frame)
                {
                    partialSolver.step(1.0f / 60.0f);
                    partialSolver.endStep();
                }
                partialSolver.waitForStep();

                const std::vector<float> vented = partialSolver.densityVolume().download();
                check(allFinite(vented), "the vented run stays finite");
                check(maxDensityInBox(vented, partial.extent, plateMin, plateMax) == 0.0,
                      "and still keeps smoke out of the plate");
                check(densityInBox(vented, partial.extent, overheadMin, overheadMax) > 10.0 * sealedOverhead,
                      "an open top lets the deflected smoke reform above the plate");

                // Back to sealed for the control below, which compares against the closed box.
                partialSolver.stepParams().openTop = false;
            }

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


        std::cout << "vorticity confinement pushes along N x omega\n";
        {
            // Fedkiw equations 9-11 on a field whose vorticity is known analytically, so the force
            // can be checked against a closed form rather than against "it looks turbulent".
            //
            // Take u = (0, f(x), 0) with f(x) = exp(-(x - xc)^2 / s). Then omega = curl(u) is
            // (0, 0, df/dx), so omega is purely along z and |omega| = |df/dx| peaks either side of
            // xc, dipping to zero at xc itself. Since |omega| varies with x alone, eta = grad|omega|
            // points along +/-x only, giving N = (+/-1, 0, 0). The force is then
            //     f_conf = eps * (N x omega) = eps * (N.y*w.z - 0, -N.x*w.z, 0) = (0, -eps*N.x*w.z, 0),
            // i.e. purely along y, with x and z components identically zero.
            //
            // Those two zeros are the discriminating check. A correct eta has no y or z component
            // on this field, so the cross product cannot produce any x or z force. The old
            // implementation differenced a *different* curl component per axis, so its eta picked
            // up spurious y and z terms here and the force tilted off the y axis.
            lr::eulerian::SmokeSolver::Config vort{};
            vort.extent           = VkExtent3D{24, 24, 24};
            vort.jacobiIterations = 0;      // pressure stays zero, so the projection is a no-op
            // Small enough that the advection preceding the force stage barely moves the seeded
            // field, so what the confinement sees is still the analytic one. Large enough that the
            // force it adds clears the noise floor.
            vort.step.dt          = 0.01f;
            vort.step.buoyancy    = 0.0f;   // no buoyancy, no source: only confinement acts
            vort.step.vorticity   = 50.0f;  // scaled up to compensate for the small dt
            vort.source.density   = 0.0f;
            vort.source.velocity[0] = 0.0f;
            vort.source.velocity[1] = 0.0f;
            vort.source.velocity[2] = 0.0f;
            vort.namePrefix       = "smokeVorticity";

            lr::eulerian::SmokeSolver vortSolver(resources, context, vort);

            const uint32_t w = vort.extent.width, h = vort.extent.height, d = vort.extent.depth;
            const double   centre  = 0.5 * static_cast<double>(w);
            const double   falloff = 8.0;

            // vy varies with x only, so omega is purely along z. The constructor's reset() already
            // cleared the field through CUDA, so the image is in the layout the kernel needs and
            // no further transition is required — the same way the obstacle cases above seed a mask.
            if (lr::eulerian::launchFillShearField(vortSolver.velocityVolume().surface(),
                                                   vortSolver.velocityVolume().extent(),
                                                   static_cast<float>(centre), static_cast<float>(falloff),
                                                   nullptr) != cudaSuccess)
                throw std::runtime_error("launchFillShearField failed");
            vortSolver.velocityVolume().synchronize();

            vortSolver.step();
            vortSolver.waitForStep();

            const std::vector<float> forced = vortSolver.velocityVolume().download();
            check(allFinite(forced), "the confinement force stays finite");

            // The same step with confinement off. Advection runs before the force stage and smears
            // the seeded field slightly, so differencing the two runs isolates the force itself
            // rather than measuring it against the analytic seed.
            lr::eulerian::SmokeSolver::Config plain = vort;
            plain.step.vorticity                    = 0.0f;
            plain.namePrefix                        = "smokeVorticityOff";
            lr::eulerian::SmokeSolver plainSolver(resources, context, plain);
            if (lr::eulerian::launchFillShearField(plainSolver.velocityVolume().surface(),
                                                   plainSolver.velocityVolume().extent(),
                                                   static_cast<float>(centre), static_cast<float>(falloff),
                                                   nullptr) != cudaSuccess)
                throw std::runtime_error("launchFillShearField failed");
            plainSolver.velocityVolume().synchronize();
            plainSolver.step();
            plainSolver.waitForStep();
            const std::vector<float> unforced = plainSolver.velocityVolume().download();

            // Interior only: the border rows are overwritten by applyBoundaries, and the one-cell
            // ring inside them has a clamped stencil, so neither reflects the force alone.
            double largestX = 0.0, largestY = 0.0, largestZ = 0.0;
            for (uint32_t z = 2; z + 2 < d; ++z)
                for (uint32_t y = 2; y + 2 < h; ++y)
                    for (uint32_t x = 2; x + 2 < w; ++x)
                    {
                        const std::size_t index = ((static_cast<std::size_t>(z) * h + y) * w + x) * 4;
                        largestX = std::max(largestX, std::fabs(static_cast<double>(forced[index]) -
                                                                static_cast<double>(unforced[index])));
                        largestY = std::max(largestY, std::fabs(static_cast<double>(forced[index + 1]) -
                                                                static_cast<double>(unforced[index + 1])));
                        largestZ = std::max(largestZ, std::fabs(static_cast<double>(forced[index + 2]) -
                                                                static_cast<double>(unforced[index + 2])));
                    }

            // N x omega is along y here, so y is where the whole force must land. x and z are the
            // discriminating checks: a mis-built eta leaks into them.
            check(largestY > 1e-4, "confinement drives flow along y, which is where N x omega points");
            check(largestX < 1e-6 * std::max(1.0, largestY),
                  "and adds no x component, since eta has no y or z part on this field");
            check(largestZ < 1e-6 * std::max(1.0, largestY),
                  "and no z component, since omega is purely along z");
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
