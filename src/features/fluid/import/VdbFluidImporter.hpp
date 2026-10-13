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
        bool        importVelocities  = true;
        float       framesPerSecond   = 24.0f;
    };

    // Imports exactly one VDB point grid as a one-frame engine-native simulation.
    static OfflineFluidSimulation importFile(const std::filesystem::path &path, const Options &options = {});
    // Lower-level entry point used by multi-file cache importers.
    static VdbFluidFrame importFrame(const std::filesystem::path &path, const Options &options = {});
};

} // namespace lr
