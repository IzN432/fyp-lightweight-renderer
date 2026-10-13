#include "features/eulerian_particle/OfflineSmokeSimulation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lr::eulerian
{

OfflineSmokeSimulation::OfflineSmokeSimulation(std::string name, glm::uvec3 gridSize, glm::vec3 voxelSize,
                                               float framesPerSecond)
    : m_name(std::move(name)), m_gridSize(gridSize), m_voxelSize(voxelSize), m_framesPerSecond(framesPerSecond)
{
    if (glm::any(glm::equal(gridSize, glm::uvec3(0))))
        throw std::invalid_argument("OfflineSmokeSimulation: grid dimensions must be positive");
    if (glm::any(glm::lessThanEqual(voxelSize, glm::vec3(0.0f))) ||
        glm::any(glm::isnan(voxelSize)) || glm::any(glm::isinf(voxelSize)))
        throw std::invalid_argument("OfflineSmokeSimulation: voxel size must be finite and positive");
    if (!std::isfinite(framesPerSecond) || framesPerSecond <= 0.0f)
        throw std::invalid_argument("OfflineSmokeSimulation: frame rate must be finite and positive");
    (void)voxelCount();
}

std::size_t OfflineSmokeSimulation::voxelCount() const
{
    uint64_t count = m_gridSize.x;
    for (uint32_t dimension : {m_gridSize.y, m_gridSize.z})
    {
        if (count > std::numeric_limits<std::size_t>::max() / dimension)
            throw std::length_error("OfflineSmokeSimulation: grid is too large");
        count *= dimension;
    }
    return static_cast<std::size_t>(count);
}

void OfflineSmokeSimulation::addFrame(float seconds, std::span<const float> density,
                                      std::span<const float> temperature, std::optional<int32_t> sourceFrame)
{
    const std::size_t count = voxelCount();
    if (density.size() != count || temperature.size() != count)
        throw std::invalid_argument("OfflineSmokeSimulation: frame field size does not match the grid");
    if (!std::isfinite(seconds) || seconds < 0.0f || (!m_frames.empty() && seconds <= m_frames.back().seconds))
        throw std::invalid_argument("OfflineSmokeSimulation: frame times must be finite and strictly increasing");
    if (std::ranges::any_of(density, [](float value) { return !std::isfinite(value); }) ||
        std::ranges::any_of(temperature, [](float value) { return !std::isfinite(value); }))
        throw std::invalid_argument("OfflineSmokeSimulation: volume fields must be finite");

    m_frames.push_back({seconds, sourceFrame.value_or(static_cast<int32_t>(m_frames.size())), m_densities.size()});
    m_densities.insert(m_densities.end(), density.begin(), density.end());
    m_temperatures.insert(m_temperatures.end(), temperature.begin(), temperature.end());
}

std::span<const float> OfflineSmokeSimulation::frameDensity(std::size_t frame) const
{
    if (frame >= m_frames.size()) throw std::out_of_range("Offline smoke frame is out of range");
    return std::span(m_densities).subspan(static_cast<std::size_t>(m_frames[frame].voxelOffset), voxelCount());
}

std::span<const float> OfflineSmokeSimulation::frameTemperature(std::size_t frame) const
{
    if (frame >= m_frames.size()) throw std::out_of_range("Offline smoke frame is out of range");
    return std::span(m_temperatures).subspan(static_cast<std::size_t>(m_frames[frame].voxelOffset), voxelCount());
}

float OfflineSmokeSimulation::durationSeconds() const
{
    return m_frames.empty() ? 0.0f : m_frames.back().seconds + 1.0f / m_framesPerSecond;
}

OfflineSmokeSimulation OfflineSmokeSimulation::fromRaw(std::string name, glm::uvec3 gridSize, glm::vec3 voxelSize,
                                                       float framesPerSecond, std::vector<Frame> frames,
                                                       std::vector<float> densities,
                                                       std::vector<float> temperatures)
{
    OfflineSmokeSimulation result(std::move(name), gridSize, voxelSize, framesPerSecond);
    result.m_frames       = std::move(frames);
    result.m_densities    = std::move(densities);
    result.m_temperatures = std::move(temperatures);
    result.validate();
    return result;
}

void OfflineSmokeSimulation::validate() const
{
    const std::size_t count = voxelCount();
    if (m_densities.size() != m_temperatures.size() || m_densities.size() != count * m_frames.size())
        throw std::invalid_argument("OfflineSmokeSimulation: field storage size mismatch");
    float previous = -1.0f;
    for (std::size_t index = 0; index < m_frames.size(); ++index)
    {
        const Frame &frame = m_frames[index];
        if (!std::isfinite(frame.seconds) || frame.seconds < 0.0f || frame.seconds <= previous ||
            frame.voxelOffset != index * count)
            throw std::invalid_argument("OfflineSmokeSimulation: invalid frame table");
        previous = frame.seconds;
    }
    if (std::ranges::any_of(m_densities, [](float value) { return !std::isfinite(value); }) ||
        std::ranges::any_of(m_temperatures, [](float value) { return !std::isfinite(value); }))
        throw std::invalid_argument("OfflineSmokeSimulation: volume fields must be finite");
}

} // namespace lr::eulerian
