#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace lr
{

// Engine-native, importer-independent representation of a baked particle simulation. Particle
// samples are stored frame-major in two contiguous arrays so a frame is both cheap to serialize
// and directly uploadable after expanding its presentation attributes.
class OfflineFluidSimulation
{
public:
    struct Frame
    {
        float     seconds        = 0.0f;
        int32_t   sourceFrame    = 0;
        uint64_t  particleOffset = 0;
        uint32_t  particleCount  = 0;
        glm::vec3 boundsMin{0.0f};
        glm::vec3 boundsMax{0.0f};

        bool operator==(const Frame &) const = default;
    };

    explicit OfflineFluidSimulation(std::string name = {}, float framesPerSecond = 24.0f);

    void addFrame(float seconds, std::span<const glm::vec3> positions,
                  std::span<const glm::vec3> velocities = {},
                  std::optional<int32_t> sourceFrame = std::nullopt);

    const std::string &name() const { return m_name; }
    void               setName(std::string name) { m_name = std::move(name); }

    float framesPerSecond() const { return m_framesPerSecond; }
    void  setFramesPerSecond(float value);
    float defaultParticleRadius() const { return m_defaultParticleRadius; }
    void  setDefaultParticleRadius(float value);

    const std::vector<Frame>     &frames() const { return m_frames; }
    const std::vector<glm::vec3> &positions() const { return m_positions; }
    const std::vector<glm::vec3> &velocities() const { return m_velocities; }
    bool                          hasVelocities() const { return !m_velocities.empty(); }

    std::span<const glm::vec3> framePositions(std::size_t frame) const;
    std::span<const glm::vec3> frameVelocities(std::size_t frame) const;
    float                      durationSeconds() const;
    uint32_t                   maximumParticleCount() const;

    // Validates data loaded from persistence without rebuilding it frame by frame.
    static OfflineFluidSimulation fromRaw(std::string name, float framesPerSecond, std::vector<Frame> frames,
                                          std::vector<glm::vec3> positions, std::vector<glm::vec3> velocities = {},
                                          float defaultParticleRadius = 0.025f);

private:
    void validate() const;

    std::string            m_name;
    float                  m_framesPerSecond       = 24.0f;
    float                  m_defaultParticleRadius = 0.025f;
    std::vector<Frame>     m_frames;
    std::vector<glm::vec3> m_positions;
    std::vector<glm::vec3> m_velocities;
};

namespace offline_fluid_examples
{

// Small deterministic clip used by tests and examples to prove the native representation without
// relying on Blender or OpenVDB. The two strands rotate while a pulse travels along the helix.
OfflineFluidSimulation makeDnaHelix(uint32_t frameCount = 48, uint32_t particlesPerStrand = 96,
                                    float framesPerSecond = 24.0f);

} // namespace offline_fluid_examples

} // namespace lr
