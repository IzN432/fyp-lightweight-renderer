#include "features/eulerian_particle/import/BlenderSmokeCacheImporter.hpp"

#include "features/fluid/import/BlenderUniImporter.hpp"

#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <map>
#include <mutex>
#include <regex>
#include <stdexcept>

namespace lr::eulerian
{
namespace
{

std::map<int32_t, std::filesystem::path> numberedFiles(const std::filesystem::path &directory,
                                                       const std::regex &pattern)
{
    std::map<int32_t, std::filesystem::path> result;
    if (!std::filesystem::is_directory(directory)) return result;
    for (const auto &entry : std::filesystem::directory_iterator(directory))
    {
        if (!entry.is_regular_file()) continue;
        std::smatch match;
        const std::string name = entry.path().filename().string();
        if (!std::regex_match(name, match, pattern)) continue;

        const std::string frameText = match[1].str();
        int32_t frame = 0;
        const auto [end, error] = std::from_chars(frameText.data(), frameText.data() + frameText.size(), frame);
        if (error != std::errc{} || end != frameText.data() + frameText.size())
            throw std::runtime_error("BlenderSmokeCacheImporter: frame number out of range in filename '" + name +
                                     "'");
        result.emplace(frame, entry.path());
    }
    return result;
}

openvdb::FloatGrid::Ptr readFloatGrid(openvdb::io::File &file, const char *name, bool required)
{
    try
    {
        auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(name));
        if (!grid) throw std::runtime_error(std::string("grid '") + name + "' is not a FloatGrid");
        return grid;
    }
    catch (const openvdb::KeyError &)
    {
        if (required) throw std::runtime_error(std::string("missing FloatGrid '") + name + "'");
        return {};
    }
}

std::size_t denseIndex(const glm::uvec3 &size, uint32_t x, uint32_t y, uint32_t z)
{
    return (static_cast<std::size_t>(z) * size.y + y) * size.x + x;
}

} // namespace

OfflineSmokeSimulation BlenderSmokeCacheImporter::importDirectory(const std::filesystem::path &cacheDirectory,
                                                                  const Options &options)
{
    static std::once_flag openVdbOnce;
    std::call_once(openVdbOnce, [] { openvdb::initialize(); });

    if (!std::filesystem::is_directory(cacheDirectory))
        throw std::runtime_error("BlenderSmokeCacheImporter: cache directory does not exist: '" +
                                 cacheDirectory.string() + "'");
    if (!std::isfinite(options.framesPerSecond) || options.framesPerSecond <= 0.0f)
        throw std::invalid_argument("BlenderSmokeCacheImporter: frame rate must be finite and positive");

    const auto dataFiles = numberedFiles(cacheDirectory / "data", std::regex(R"(fluid_data_(-?\d+)\.vdb)"));
    const auto configFiles = numberedFiles(cacheDirectory / "config", std::regex(R"(config_(-?\d+)\.uni)"));
    if (dataFiles.empty())
        throw std::runtime_error("BlenderSmokeCacheImporter: no data/fluid_data_####.vdb files found");

    std::optional<OfflineSmokeSimulation> result;
    std::optional<int32_t> firstImported;
    for (const auto &[frameNumber, dataPath] : dataFiles)
    {
        if ((options.firstFrame && frameNumber < *options.firstFrame) ||
            (options.lastFrame && frameNumber > *options.lastFrame))
            continue;
        const auto config = configFiles.find(frameNumber);
        if (config == configFiles.end())
            throw std::runtime_error("BlenderSmokeCacheImporter: missing config/config_" +
                                     std::to_string(frameNumber) + ".uni");

        const BlenderFluidDomainConfig domain = BlenderUniImporter::importFile(config->second);
        if (glm::any(glm::lessThanEqual(domain.resolution, glm::ivec3(0))))
            throw std::runtime_error("BlenderSmokeCacheImporter: invalid domain resolution");

        openvdb::io::File file(dataPath.string());
        try
        {
            file.open();
            const auto densityGrid = readFloatGrid(file, "density", true);
            const auto temperatureGrid = readFloatGrid(file, "temperature", false);

            const glm::uvec3 blenderSize(domain.resolution);
            const glm::uvec3 engineSize(blenderSize.x, blenderSize.z, blenderSize.y);
            const openvdb::Vec3d voxel = densityGrid->voxelSize();
            const glm::vec3 engineVoxel(static_cast<float>(voxel.x()), static_cast<float>(voxel.z()),
                                        static_cast<float>(voxel.y()));
            const float voxelTolerance = 1e-5f * std::max({engineVoxel.x, engineVoxel.y, engineVoxel.z, 1.0f});
            if (std::max({std::abs(engineVoxel.x - engineVoxel.y), std::abs(engineVoxel.x - engineVoxel.z),
                          std::abs(engineVoxel.y - engineVoxel.z)}) > voxelTolerance)
                throw std::runtime_error("BlenderSmokeCacheImporter: non-cubic VDB voxels are not supported");
            if (!result)
                result.emplace(cacheDirectory.filename().string(), engineSize, engineVoxel, options.framesPerSecond);
            else if (result->gridSize() != engineSize ||
                     glm::any(glm::greaterThan(glm::abs(result->voxelSize() - engineVoxel), glm::vec3(1e-5f))))
                throw std::runtime_error("BlenderSmokeCacheImporter: the grid size or voxel transform changes between frames");

            const std::size_t count = static_cast<std::size_t>(engineSize.x) * engineSize.y * engineSize.z;
            std::vector<float> density(count, 0.0f);
            std::vector<float> temperature(count, 0.0f);
            for (uint32_t bz = 0; bz < blenderSize.z; ++bz)
                for (uint32_t by = 0; by < blenderSize.y; ++by)
                    for (uint32_t bx = 0; bx < blenderSize.x; ++bx)
                    {
                        const openvdb::Coord coordinate(static_cast<int>(bx) + domain.resolutionMin.x,
                                                        static_cast<int>(by) + domain.resolutionMin.y,
                                                        static_cast<int>(bz) + domain.resolutionMin.z);
                        const uint32_t ex = bx;
                        const uint32_t ey = bz;
                        const uint32_t ez = blenderSize.y - 1u - by;
                        const std::size_t index = denseIndex(engineSize, ex, ey, ez);
                        // This is currently a smoke renderer, not a fire renderer. Keep Blender's
                        // flame channel out of both extinction and colour instead of disguising
                        // emissive fire as extra smoke density and temperature.
                        density[index] = std::max(densityGrid->tree().getValue(coordinate), 0.0f);
                        temperature[index] =
                            std::max(temperatureGrid ? temperatureGrid->tree().getValue(coordinate) : 0.0f, 0.0f);
                    }
            file.close();

            if (!firstImported) firstImported = frameNumber;
            result->addFrame(static_cast<float>(frameNumber - *firstImported) / options.framesPerSecond,
                             density, temperature, frameNumber);
        }
        catch (const std::exception &error)
        {
            try { file.close(); } catch (...) {}
            throw std::runtime_error("BlenderSmokeCacheImporter: failed to import '" + dataPath.string() + "': " +
                                     error.what());
        }
    }
    if (!result || result->frames().empty())
        throw std::runtime_error("BlenderSmokeCacheImporter: selected frame range contains no gas frames");
    return std::move(*result);
}

} // namespace lr::eulerian
