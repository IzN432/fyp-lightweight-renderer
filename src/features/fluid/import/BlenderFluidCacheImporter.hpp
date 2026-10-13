#pragma once

#include "features/fluid/OfflineFluidSimulation.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>

namespace lr
{

class BlenderFluidCacheImporter
{
public:
    struct Options
    {
        float                  framesPerSecond = 24.0f;
        std::optional<int32_t> firstFrame;
        std::optional<int32_t> lastFrame;
        bool                   importVelocities = true;
        // Converts VDB coordinates with origin at the domain's lower corner into Blender domain-
        // local coordinates centred on the object origin.
        bool centerDomain = true;
    };

    static OfflineFluidSimulation importDirectory(const std::filesystem::path &cacheDirectory,
                                                  const Options               &options = {});
};

} // namespace lr
