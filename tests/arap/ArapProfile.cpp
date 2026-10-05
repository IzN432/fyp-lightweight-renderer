#include "core/loaders/GltfLoader.hpp"
#include "core/loaders/MaterialStore.hpp"
#include "features/arap/ArapSolver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace
{

struct Options
{
    int    iterations         = 10;
    int    trials             = 10;
    double anchorBandFraction = 0.02;
    double handleOffsetFraction = 0.10;
};

int parsePositiveInt(const char *value, const char *name)
{
    const int parsed = std::stoi(value);
    if (parsed <= 0)
    {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
    return parsed;
}

double parseFraction(const char *value, const char *name)
{
    const double parsed = std::stod(value);
    if (!(parsed > 0.0 && parsed < 1.0))
    {
        throw std::runtime_error(std::string(name) + " must be between 0 and 1");
    }
    return parsed;
}

Options parseOptions(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        auto requireValue = [&](const char *name) {
            if (++i >= argc)
            {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[i];
        };

        if (argument == "--iterations")
        {
            options.iterations = parsePositiveInt(requireValue("--iterations"), "--iterations");
        } else if (argument == "--trials")
        {
            options.trials = parsePositiveInt(requireValue("--trials"), "--trials");
        } else if (argument == "--anchor-band")
        {
            options.anchorBandFraction = parseFraction(requireValue("--anchor-band"), "--anchor-band");
        } else if (argument == "--handle-offset")
        {
            options.handleOffsetFraction = parseFraction(requireValue("--handle-offset"), "--handle-offset");
        } else if (argument == "--help")
        {
            std::cout << "Usage: arap_profile [--iterations N] [--trials N] "
                         "[--anchor-band FRACTION] [--handle-offset FRACTION]\n";
            std::exit(0);
        } else
        {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    return options;
}

double median(std::vector<double> samples)
{
    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    return samples.size() % 2 == 0 ? (samples[middle - 1] + samples[middle]) * 0.5 : samples[middle];
}

} // namespace

int main(int argc, char **argv)
try
{
    const Options options = parseOptions(argc, argv);

    lr::MaterialStore materials(64, [] { return lr::Material{}; });
    auto loaded = lr::GltfLoader::load(
        std::filesystem::path(LR_SAMPLE_ASSET_DIR) / "lion_head_4k.glb", materials);
    if (loaded.sequence.frames.empty())
    {
        throw std::runtime_error("lion GLB contains no meshes");
    }
    if (loaded.sequence.frames.size() != 1)
    {
        throw std::runtime_error("lion benchmark expects exactly one mesh");
    }

    const lr::Mesh &mesh = loaded.sequence.frames.front();
    if (mesh.positions().empty())
    {
        throw std::runtime_error("lion mesh contains no positions");
    }

    glm::vec3 boundsMin(std::numeric_limits<float>::max());
    glm::vec3 boundsMax(std::numeric_limits<float>::lowest());
    uint32_t  handle = 0;
    for (uint32_t i = 0; i < mesh.positions().size(); ++i)
    {
        const glm::vec3 &position = mesh.positions()[i];
        boundsMin = glm::min(boundsMin, position);
        boundsMax = glm::max(boundsMax, position);
        if (position.y > mesh.positions()[handle].y)
        {
            handle = i;
        }
    }

    const float height = boundsMax.y - boundsMin.y;
    if (!(height > 0.0f))
    {
        throw std::runtime_error("lion mesh has zero Y extent");
    }

    const float anchorCeiling = boundsMin.y + height * static_cast<float>(options.anchorBandFraction);
    std::vector<uint32_t> anchors;
    for (uint32_t i = 0; i < mesh.positions().size(); ++i)
    {
        if (i != handle && mesh.positions()[i].y <= anchorCeiling)
        {
            anchors.push_back(i);
        }
    }
    if (anchors.empty())
    {
        throw std::runtime_error("anchor band selected no pedestal vertices");
    }

    const std::vector<uint32_t> handles{handle};
    lr::ArapSolver solver;
    if (!solver.precompute(mesh, anchors, handles))
    {
        throw std::runtime_error("ARAP precomputation failed");
    }

    const std::vector<glm::vec3> restPositions = mesh.positions();
    const glm::vec3 handleTarget = restPositions[handle] +
                                   glm::vec3(height * static_cast<float>(options.handleOffsetFraction), 0.0f, 0.0f);
    const std::unordered_map<uint32_t, glm::vec3> targets{{handle, handleTarget}};

    // Warm caches and lazy library state without including that run in the samples.
    (void)solver.solve(targets, restPositions, options.iterations);

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(options.trials));
    for (int trial = 0; trial < options.trials; ++trial)
    {
        (void)solver.solve(targets, restPositions, options.iterations);
        const double milliseconds = solver.performanceStats().lastSolveMs;
        samples.push_back(milliseconds);
        std::cout << "trial " << std::setw(2) << trial + 1 << ": " << std::fixed << std::setprecision(3)
                  << milliseconds << " ms\n";
    }

    const auto [minimum, maximum] = std::minmax_element(samples.begin(), samples.end());
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        static_cast<double>(samples.size());

    std::cout << "\nlion ARAP profile\n"
              << "  vertices:       " << mesh.positions().size() << '\n'
              << "  triangles:      " << mesh.faces().size() << '\n'
              << "  handle:         " << handle << " at y=" << restPositions[handle].y << '\n'
              << "  anchors:        " << anchors.size() << " at y <= " << anchorCeiling << " (bottom "
              << options.anchorBandFraction * 100.0 << "%)\n"
              << "  displacement:   +" << options.handleOffsetFraction * 100.0 << "% height on X\n"
              << "  iterations:     " << options.iterations << '\n'
              << "  precompute:     " << solver.performanceStats().precomputeMs << " ms\n"
              << "  solve median:   " << median(samples) << " ms total\n"
              << "  solve mean:     " << mean << " ms total\n"
              << "  solve range:    " << *minimum << " .. " << *maximum << " ms\n"
              << "  worst case framerate: " << 1000.0 / *maximum << " fps\n"
              << "  median/iter:    " << median(samples) / static_cast<double>(options.iterations) << " ms\n";
    return 0;
} catch (const std::exception &error)
{
    std::cerr << "arap_profile: " << error.what() << '\n';
    return 1;
}
