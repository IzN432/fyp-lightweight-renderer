#pragma once

#include "features/fluid/OfflineFluidSimulation.hpp"

#include <glm/glm.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace lr
{

struct VdbFluidFrame
{
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> velocities;
    glm::vec3              voxelSize{1.0f};
};

class VdbFluidImporter
{
public:
    struct Options
    {
        std::string pointGridName     = "particles";
        std::string velocityAttribute = "particles_velocity";
        // Off by default, matching BlenderFluidCacheImporter: the channel is groundwork for frame
        // interpolation, and nothing reads it for rendering today.
        bool        importVelocities  = false;
        float       framesPerSecond   = 24.0f;
    };

    // Imports exactly one VDB point grid as a one-frame engine-native simulation.
    //
    // Coordinates come back in the file's own axes. OpenVDB has no up-axis convention, so this
    // reader deliberately applies no rotation: a VDB written by a Z-up tool imports lying on its
    // side here, and it is the format-specific importer that knows which convention its files
    // follow (BlenderFluidCacheImporter::Options::convertUpAxis does this for Blender caches).
    // Importing a Blender cache through this entry point therefore skips that conversion.
    static OfflineFluidSimulation importFile(const std::filesystem::path &path, const Options &options = {});
    // Lower-level entry point used by multi-file cache importers. Also in the file's own axes.
    static VdbFluidFrame importFrame(const std::filesystem::path &path, const Options &options = {});
};

} // namespace lr
