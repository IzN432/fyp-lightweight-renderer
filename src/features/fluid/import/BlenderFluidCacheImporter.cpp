#include "features/fluid/import/BlenderFluidCacheImporter.hpp"

#include "features/fluid/import/BlenderUniImporter.hpp"
#include "features/fluid/import/VdbFluidImporter.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>

namespace lr
{
namespace
{

std::map<int32_t, std::filesystem::path> numberedFiles(const std::filesystem::path &directory,
                                                       const std::regex            &pattern)
{
    std::map<int32_t, std::filesystem::path> result;
    if (!std::filesystem::is_directory(directory))
    {
        return result;
    }
    for (const auto &entry : std::filesystem::directory_iterator(directory))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        std::smatch       match;
        const std::string name = entry.path().filename().string();
        if (!std::regex_match(name, match, pattern))
        {
            continue;
        }
        result.emplace(std::stoi(match[1].str()), entry.path());
    }
    return result;
}

} // namespace

OfflineFluidSimulation BlenderFluidCacheImporter::importDirectory(const std::filesystem::path &cacheDirectory,
                                                                  const Options               &options)
{
    if (!std::filesystem::is_directory(cacheDirectory))
    {
        throw std::runtime_error("BlenderFluidCacheImporter: cache directory does not exist: '" +
                                 cacheDirectory.string() + "'");
    }
    if (!std::isfinite(options.framesPerSecond) || options.framesPerSecond <= 0.0f)
    {
        throw std::invalid_argument("BlenderFluidCacheImporter: frame rate must be finite and positive");
    }

    const auto dataFiles   = numberedFiles(cacheDirectory / "data", std::regex(R"(fluid_data_(-?\d+)\.vdb)"));
    const auto configFiles = numberedFiles(cacheDirectory / "config", std::regex(R"(config_(-?\d+)\.uni)"));
    if (dataFiles.empty())
    {
        throw std::runtime_error("BlenderFluidCacheImporter: no data/fluid_data_####.vdb files found");
    }

    OfflineFluidSimulation    result(cacheDirectory.filename().string(), options.framesPerSecond);
    std::optional<int32_t>    firstImported;
    VdbFluidImporter::Options vdbOptions;
    vdbOptions.importVelocities = options.importVelocities;

    for (const auto &[frameNumber, dataPath] : dataFiles)
    {
        if (options.firstFrame && frameNumber < *options.firstFrame)
        {
            continue;
        }
        if (options.lastFrame && frameNumber > *options.lastFrame)
        {
            continue;
        }
        const auto config = configFiles.find(frameNumber);
        if (config == configFiles.end())
        {
            throw std::runtime_error("BlenderFluidCacheImporter: missing config/config_" + std::to_string(frameNumber) +
                                     ".uni");
        }

        const BlenderFluidDomainConfig domain = BlenderUniImporter::importFile(config->second);
        VdbFluidFrame                  frame  = VdbFluidImporter::importFrame(dataPath, vdbOptions);
        if (frame.positions.empty())
        {
            continue;
        }
        if (!firstImported)
        {
            result.setDefaultParticleRadius(0.5f * std::min({frame.voxelSize.x, frame.voxelSize.y, frame.voxelSize.z}));
        }
        if (options.centerDomain)
        {
            const glm::vec3 domainExtent = glm::vec3(domain.baseResolution) * frame.voxelSize;
            const glm::vec3 centre       = domainExtent * 0.5f;
            for (glm::vec3 &position : frame.positions)
            {
                position -= centre;
            }
        }
        if (!frame.velocities.empty())
        {
            // Mantaflow particle velocities are in cells per solver time while point positions are
            // exported in VDB world units. Convert the optional channel to the same spatial units.
            for (glm::vec3 &velocity : frame.velocities)
            {
                velocity *= frame.voxelSize;
            }
        }
        if (!firstImported)
        {
            firstImported = frameNumber;
        }
        const float seconds = static_cast<float>(frameNumber - *firstImported) / options.framesPerSecond;
        result.addFrame(seconds, frame.positions, frame.velocities, frameNumber);
    }
    if (result.frames().empty())
    {
        throw std::runtime_error("BlenderFluidCacheImporter: selected frame range contains no particles");
    }
    return result;
}

} // namespace lr
