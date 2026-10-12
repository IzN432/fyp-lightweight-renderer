#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/framegraph/ResourceRegistry.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace lr
{

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
        float       radius                      = 0.04f;
        glm::vec3   color{0.08f, 0.45f, 1.0f};
    };

    ParticleBillboardPass(Config config, ResourceRegistry &resources);

    void build(FrameGraph &frameGraph);
    void setPositions(std::span<const glm::vec3> positions);
    void setStyle(float radius, const glm::vec3 &color)
    {
        m_config.radius = radius;
        m_config.color  = color;
    }

    uint32_t particleCount() const { return m_particleCount; }

private:
    Config            m_config;
    ResourceRegistry *m_resources = nullptr;
    uint32_t          m_particleCount = 0;
    std::vector<glm::vec4> m_upload;
};

} // namespace lr
