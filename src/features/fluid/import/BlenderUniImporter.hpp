#pragma once

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

namespace lr
{

struct BlenderFluidDomainConfig
{
    int32_t             activeFields = 0;
    glm::ivec3          resolution{0};
    float               cellSize = 0.0f;
    float               timeStep = 0.0f;
    glm::vec3           boundsMin{0.0f};
    glm::vec3           boundsMax{0.0f};
    glm::vec3           previousBoundsMin{0.0f};
    glm::ivec3          shift{0};
    glm::vec3           objectShift{0.0f};
    glm::mat4           objectTransform{1.0f};
    glm::ivec3          baseResolution{0};
    glm::ivec3          resolutionMin{0};
    glm::ivec3          resolutionMax{0};
    glm::vec3           activeColor{0.0f};
    float               totalTime = 0.0f;
    std::array<char, 4> version{};

    std::string versionString() const;
};

class BlenderUniImporter
{
public:
    // Reads Blender's 204-byte gzip-compressed per-frame Mantaflow configuration record.
    static BlenderFluidDomainConfig importFile(const std::filesystem::path &path);
};

} // namespace lr
