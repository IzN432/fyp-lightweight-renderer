#pragma once

#include "features/eulerian_particle/OfflineSmokeSimulation.hpp"

#include <filesystem>
#include <optional>

namespace lr::eulerian
{

class BlenderSmokeCacheImporter
{
public:
    struct Options
    {
        float                  framesPerSecond = 24.0f;
        std::optional<int32_t> firstFrame;
        std::optional<int32_t> lastFrame;
    };

    static OfflineSmokeSimulation importDirectory(const std::filesystem::path &cacheDirectory,
                                                  const Options &options = {});
};

} // namespace lr::eulerian
