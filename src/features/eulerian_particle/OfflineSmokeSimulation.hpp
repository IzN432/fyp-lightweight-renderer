#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace lr::eulerian
{

// Engine-native baked gas clip. Frames share one fixed, dense grid so their scalar fields can be
// copied directly into the CUDA/Vulkan volumes used by SmokeRenderPass.
class OfflineSmokeSimulation
{
public:
    struct Frame
    {
        float    seconds     = 0.0f;
        int32_t  sourceFrame = 0;
        uint64_t voxelOffset = 0;

        bool operator==(const Frame &) const = default;
    };

    OfflineSmokeSimulation(std::string name = {}, glm::uvec3 gridSize = glm::uvec3(1),
                           glm::vec3 voxelSize = glm::vec3(1.0f), float framesPerSecond = 24.0f);

    void addFrame(float seconds, std::span<const float> density, std::span<const float> temperature,
                  std::optional<int32_t> sourceFrame = std::nullopt);

    const std::string &name() const { return m_name; }
    glm::uvec3         gridSize() const { return m_gridSize; }
    glm::vec3          voxelSize() const { return m_voxelSize; }
    glm::vec3          worldSize() const { return glm::vec3(m_gridSize) * m_voxelSize; }
    float              framesPerSecond() const { return m_framesPerSecond; }
    float              durationSeconds() const;

    const std::vector<Frame> &frames() const { return m_frames; }
    const std::vector<float> &densities() const { return m_densities; }
    const std::vector<float> &temperatures() const { return m_temperatures; }
    std::span<const float>    frameDensity(std::size_t frame) const;
    std::span<const float>    frameTemperature(std::size_t frame) const;

    static OfflineSmokeSimulation fromRaw(std::string name, glm::uvec3 gridSize, glm::vec3 voxelSize,
                                          float framesPerSecond, std::vector<Frame> frames,
                                          std::vector<float> densities, std::vector<float> temperatures);

private:
    std::size_t voxelCount() const;
    void        validate() const;

    std::string        m_name;
    glm::uvec3         m_gridSize{1};
    glm::vec3          m_voxelSize{1.0f};
    float              m_framesPerSecond = 24.0f;
    std::vector<Frame> m_frames;
    std::vector<float> m_densities;
    std::vector<float> m_temperatures;
};

} // namespace lr::eulerian
