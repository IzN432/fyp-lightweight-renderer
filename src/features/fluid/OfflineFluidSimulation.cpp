#include "features/fluid/OfflineFluidSimulation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lr
{
namespace
{

void requireFinite(const glm::vec3 &value, const char *what)
{
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
    {
        throw std::invalid_argument(std::string("OfflineFluidSimulation: non-finite ") + what);
    }
}

} // namespace

OfflineFluidSimulation::OfflineFluidSimulation(std::string name, float framesPerSecond) : m_name(std::move(name))
{
    setFramesPerSecond(framesPerSecond);
}

void OfflineFluidSimulation::setFramesPerSecond(float value)
{
    if (!std::isfinite(value) || value <= 0.0f)
    {
        throw std::invalid_argument("OfflineFluidSimulation: frame rate must be finite and positive");
    }
    m_framesPerSecond = value;
}

void OfflineFluidSimulation::setDefaultParticleRadius(float value)
{
    if (!std::isfinite(value) || value <= 0.0f)
    {
        throw std::invalid_argument("OfflineFluidSimulation: default particle radius must be finite and positive");
    }
    m_defaultParticleRadius = value;
}

void OfflineFluidSimulation::addFrame(float seconds, std::span<const glm::vec3> positions,
                                      std::span<const glm::vec3> velocities,
                                      std::optional<int32_t> sourceFrame)
{
    if (!std::isfinite(seconds) || seconds < 0.0f)
    {
        throw std::invalid_argument("OfflineFluidSimulation: frame time must be finite and non-negative");
    }
    if (!m_frames.empty() && seconds <= m_frames.back().seconds)
    {
        throw std::invalid_argument("OfflineFluidSimulation: frame times must be strictly increasing");
    }
    if (!velocities.empty() && velocities.size() != positions.size())
    {
        throw std::invalid_argument("OfflineFluidSimulation: a velocity is required for every particle");
    }
    if (positions.size() > std::numeric_limits<uint32_t>::max())
    {
        throw std::length_error("OfflineFluidSimulation: frame particle count exceeds uint32_t");
    }
    if (!m_frames.empty() && hasVelocities() != !velocities.empty())
    {
        throw std::invalid_argument("OfflineFluidSimulation: velocity availability must be consistent across frames");
    }

    glm::vec3 boundsMin(0.0f);
    glm::vec3 boundsMax(0.0f);
    if (!positions.empty())
    {
        boundsMin = positions.front();
        boundsMax = positions.front();
        for (const glm::vec3 &position : positions)
        {
            requireFinite(position, "particle position");
            boundsMin = glm::min(boundsMin, position);
            boundsMax = glm::max(boundsMax, position);
        }
    }
    for (const glm::vec3 &velocity : velocities)
    {
        requireFinite(velocity, "particle velocity");
    }

    const uint64_t offset = m_positions.size();
    const int32_t resolvedSourceFrame = sourceFrame.value_or(static_cast<int32_t>(m_frames.size()));
    m_frames.push_back(
        {seconds, resolvedSourceFrame, offset, static_cast<uint32_t>(positions.size()), boundsMin, boundsMax});
    m_positions.insert(m_positions.end(), positions.begin(), positions.end());
    m_velocities.insert(m_velocities.end(), velocities.begin(), velocities.end());
}

std::span<const glm::vec3> OfflineFluidSimulation::framePositions(std::size_t frame) const
{
    if (frame >= m_frames.size())
    {
        throw std::out_of_range("Offline fluid frame is out of range");
    }
    const Frame &entry = m_frames[frame];
    return std::span(m_positions).subspan(static_cast<std::size_t>(entry.particleOffset), entry.particleCount);
}

std::span<const glm::vec3> OfflineFluidSimulation::frameVelocities(std::size_t frame) const
{
    if (!hasVelocities())
    {
        return {};
    }
    if (frame >= m_frames.size())
    {
        throw std::out_of_range("Offline fluid frame is out of range");
    }
    const Frame &entry = m_frames[frame];
    return std::span(m_velocities).subspan(static_cast<std::size_t>(entry.particleOffset), entry.particleCount);
}

float OfflineFluidSimulation::durationSeconds() const
{
    if (m_frames.empty())
    {
        return 0.0f;
    }
    return m_frames.back().seconds + 1.0f / m_framesPerSecond;
}

uint32_t OfflineFluidSimulation::maximumParticleCount() const
{
    uint32_t maximum = 0;
    for (const Frame &frame : m_frames)
    {
        maximum = std::max(maximum, frame.particleCount);
    }
    return maximum;
}

OfflineFluidSimulation OfflineFluidSimulation::fromRaw(std::string name, float framesPerSecond,
                                                       std::vector<Frame> frames, std::vector<glm::vec3> positions,
                                                       std::vector<glm::vec3> velocities, float defaultParticleRadius)
{
    OfflineFluidSimulation result(std::move(name), framesPerSecond);
    result.setDefaultParticleRadius(defaultParticleRadius);
    result.m_frames     = std::move(frames);
    result.m_positions  = std::move(positions);
    result.m_velocities = std::move(velocities);
    result.validate();
    return result;
}

void OfflineFluidSimulation::validate() const
{
    uint64_t expectedOffset = 0;
    float    previousTime   = -1.0f;
    for (const Frame &frame : m_frames)
    {
        if (!std::isfinite(frame.seconds) || frame.seconds < 0.0f || frame.seconds <= previousTime)
        {
            throw std::invalid_argument("OfflineFluidSimulation: invalid frame time table");
        }
        if (frame.particleOffset != expectedOffset || expectedOffset > m_positions.size() ||
            frame.particleCount > m_positions.size() - expectedOffset)
        {
            throw std::invalid_argument("OfflineFluidSimulation: invalid frame particle range");
        }
        requireFinite(frame.boundsMin, "frame bounds");
        requireFinite(frame.boundsMax, "frame bounds");
        expectedOffset += frame.particleCount;
        previousTime = frame.seconds;
    }
    if (expectedOffset != m_positions.size())
    {
        throw std::invalid_argument("OfflineFluidSimulation: unreferenced particle samples");
    }
    if (!m_velocities.empty() && m_velocities.size() != m_positions.size())
    {
        throw std::invalid_argument("OfflineFluidSimulation: velocity sample count mismatch");
    }
    for (const glm::vec3 &position : m_positions)
    {
        requireFinite(position, "particle position");
    }
    for (const glm::vec3 &velocity : m_velocities)
    {
        requireFinite(velocity, "particle velocity");
    }
}

namespace offline_fluid_examples
{

OfflineFluidSimulation makeDnaHelix(uint32_t frameCount, uint32_t particlesPerStrand, float framesPerSecond)
{
    if (frameCount == 0 || particlesPerStrand < 2)
    {
        throw std::invalid_argument("DNA helix example requires frames and at least two particles per strand");
    }

    OfflineFluidSimulation result("DNA helix", framesPerSecond);
    std::vector<glm::vec3> positions;
    positions.reserve(static_cast<std::size_t>(particlesPerStrand) * 2);
    constexpr float tau = 6.28318530717958647692f;
    for (uint32_t frame = 0; frame < frameCount; ++frame)
    {
        positions.clear();
        const float phase = tau * static_cast<float>(frame) / static_cast<float>(frameCount);
        for (uint32_t strand = 0; strand < 2; ++strand)
        {
            const float strandPhase = static_cast<float>(strand) * 0.5f * tau;
            for (uint32_t particle = 0; particle < particlesPerStrand; ++particle)
            {
                const float t     = static_cast<float>(particle) / static_cast<float>(particlesPerStrand - 1);
                const float angle = 4.0f * tau * t + phase + strandPhase;
                const float pulse = 1.0f + 0.08f * std::sin(2.0f * tau * t - phase * 2.0f);
                positions.emplace_back(0.35f * pulse * std::cos(angle), 1.6f * (t - 0.5f),
                                       0.35f * pulse * std::sin(angle));
            }
        }
        result.addFrame(static_cast<float>(frame) / framesPerSecond, positions);
    }
    return result;
}

} // namespace offline_fluid_examples
} // namespace lr
