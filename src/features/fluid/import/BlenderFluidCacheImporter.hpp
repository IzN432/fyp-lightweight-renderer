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
        // Off by default: nothing renders the velocity channel yet (baked playback uploads
        // positions only), so importing it would double clip memory and .lrscene size for data
        // only a future frame-interpolation mode would read. Turn it on when that lands.
        bool                   importVelocities = false;
        // Converts VDB coordinates with origin at the domain's lower corner into Blender domain-
        // local coordinates centred on the object origin.
        bool centerDomain = true;
        // Rotates Blender's Z-up coordinates into this engine's Y-up convention, for positions and
        // velocities alike. Off imports the cache in Blender's own axes, which is only useful for
        // inspecting a cache as authored -- a clip imported that way renders lying on its side.
        bool convertUpAxis = true;
    };

    static OfflineFluidSimulation importDirectory(const std::filesystem::path &cacheDirectory,
                                                  const Options               &options = {});
};

} // namespace lr
