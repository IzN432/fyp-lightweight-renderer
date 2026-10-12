#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/framegraph/ResourceRegistry.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <span>
#include <type_traits>
#include <string>

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

// Debug/reference particle renderer. Each particle is expanded to a camera-facing quad in the
// vertex shader; the fragment shader turns it into a sphere impostor and writes corrected depth.
class ParticleBillboardPass
{
public:
    struct Config
    {
        std::string cameraBufferResourceName;
        std::string particleBufferResourceName = "fluid_particles";
        uint32_t    maxParticles                = 65536;
    };

    ParticleBillboardPass(Config config, ResourceRegistry &resources);

    void build(FrameGraph &frameGraph);
    void setParticles(std::span<const ParticleGpuData> particles);

    uint32_t particleCount() const { return m_particleCount; }

private:
    Config            m_config;
    ResourceRegistry *m_resources = nullptr;
    uint32_t          m_particleCount = 0;
};

} // namespace lr
