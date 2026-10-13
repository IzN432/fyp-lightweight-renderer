#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/framegraph/ResourceRegistry.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace lr
{

enum class ParticleType : uint32_t
{
    Liquid = 0,
    Foam   = 1,
    Spray  = 2,
    Bubble = 3,
};

// Stable CPU/GPU upload contract. Two vec4s keep every record naturally aligned for a std430
// storage-buffer array. The type is encoded as a float because the second channel is deliberately
// a vec4; the small integral values above are represented exactly.
struct alignas(16) ParticleGpuData
{
    glm::vec4 positionAndRadius{0.0f};
    glm::vec4 colorAndType{1.0f, 1.0f, 1.0f, static_cast<float>(ParticleType::Liquid)};
};

static_assert(sizeof(ParticleGpuData) == 32);
static_assert(alignof(ParticleGpuData) == 16);
static_assert(std::is_standard_layout_v<ParticleGpuData>);

inline ParticleGpuData makeParticleGpuData(const glm::vec3 &position, float radius, const glm::vec3 &color,
                                           ParticleType type = ParticleType::Liquid)
{
    return {
        .positionAndRadius = glm::vec4(position, radius),
        .colorAndType      = glm::vec4(color, static_cast<float>(type)),
    };
}

namespace particle_billboard
{

// CPU reference for the vertex shader's projected half-size. Keeping this beside the upload
// contract gives tests a GPU-free way to lock down the projection convention used by the GLSL.
inline glm::vec2 projectedRadiusNdc(const glm::mat4 &projection, const glm::vec3 &viewCenter, float radius)
{
    const glm::vec4 centerClip = projection * glm::vec4(viewCenter, 1.0f);
    const glm::vec4 xClip      = projection * glm::vec4(viewCenter + glm::vec3(radius, 0.0f, 0.0f), 1.0f);
    const glm::vec4 yClip      = projection * glm::vec4(viewCenter + glm::vec3(0.0f, radius, 0.0f), 1.0f);
    const glm::vec2 centerNdc  = glm::vec2(centerClip) / centerClip.w;
    return glm::abs(glm::vec2(xClip.x / xClip.w - centerNdc.x, yClip.y / yClip.w - centerNdc.y));
}

// Reconstructs the visible hemisphere from the quad-local coordinate used by the fragment shader.
// View space looks down -Z, so the visible surface is displaced toward +Z from the centre.
inline std::optional<glm::vec3> reconstructSurface(const glm::vec3 &viewCenter, float radius,
                                                   const glm::vec2 &sphereCoord)
{
    const float radiusSquared = glm::dot(sphereCoord, sphereCoord);
    if (radius <= 0.0f || radiusSquared > 1.0f)
    {
        return std::nullopt;
    }
    const float nearZ = std::sqrt(std::max(0.0f, 1.0f - radiusSquared));
    return viewCenter + glm::vec3(sphereCoord, nearZ) * radius;
}

inline float projectDepth(const glm::mat4 &projection, const glm::vec3 &viewPosition)
{
    const glm::vec4 clip = projection * glm::vec4(viewPosition, 1.0f);
    return clip.z / clip.w;
}

inline uint32_t checkedParticleCount(std::size_t count, uint32_t capacity)
{
    if (count > capacity)
    {
        throw std::length_error("ParticleBillboardPass particle count exceeds configured capacity");
    }
    return static_cast<uint32_t>(count);
}

} // namespace particle_billboard

// Debug/reference particle renderer. Each particle is expanded to a camera-facing quad in the
// vertex shader; the fragment shader turns it into a sphere impostor and writes corrected depth.
class ParticleBillboardPass
{
public:
    struct Config
    {
        std::string cameraBufferResourceName;
        std::string particleBufferResourceName = "fluid_particles";
        uint32_t    maxParticles               = 65536;
        // Whether this pass allocates the particle buffer. False means another component already
        // registered it and this pass only reads it -- which is how the GPU solver feeds the
        // renderer: it writes the records in a compute pass and nothing is uploaded from the host,
        // so the buffer has to be the solver's device-local one rather than a dynamic buffer this
        // pass would own. setParticles() is then unavailable, because there is nothing to upload.
        bool ownsParticleBuffer = true;
    };

    ParticleBillboardPass(Config config, ResourceRegistry &resources);

    void build(FrameGraph &frameGraph);
    // Uploads records from the host. Only valid when the pass owns its particle buffer; with an
    // externally written one (see Config::ownsParticleBuffer) the producer sets the count through
    // setParticleCount() instead.
    void setParticles(std::span<const ParticleGpuData> particles);
    // How many records of an externally written buffer to draw.
    void setParticleCount(uint32_t count);

    uint32_t particleCount() const { return m_particleCount; }

private:
    Config            m_config;
    ResourceRegistry *m_resources     = nullptr;
    uint32_t          m_particleCount = 0;
};

} // namespace lr
