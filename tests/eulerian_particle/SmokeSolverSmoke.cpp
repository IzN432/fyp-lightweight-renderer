#include "core/framegraph/ResourceRegistry.hpp"
#include "core/vulkan/Allocator.hpp"
#include "core/vulkan/VulkanContext.hpp"
#include "features/eulerian_particle/SmokeSolver.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
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

} // namespace

int main()
{
    try
    {
        std::vector<const char *> externalMemoryExtensions = {VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME};
#if defined(_WIN32)
        externalMemoryExtensions.push_back("VK_KHR_external_memory_win32");
#elif defined(__linux__)
        externalMemoryExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
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
        for (int frame = 0; frame < 60; ++frame)
            solver.step();
        const std::vector<float> density60 = solver.densityVolume().download();
        check(allFinite(density60), "density stays finite over 60 steps");
        check(allFinite(solver.velocityVolume().download()), "velocity stays finite over 60 steps");
        check(allFinite(solver.pressureVolume().download()), "pressure stays finite over 60 steps");
        check(centroidY(density60) > startHeight, "the density centroid rises");

        std::cout << "an explicit timestep does not disturb the configured one\n";
        solver.step(1.0f / 120.0f);
        check(solver.config().step.dt == config.step.dt, "step(dt) restores Config::step.dt");

        std::cout << "reset clears the simulation again\n";
        solver.reset();
        check(maxAbs(solver.densityVolume().download()) == 0.0, "reset() empties a running simulation");

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
