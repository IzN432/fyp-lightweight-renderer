#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/framegraph/ResourceRegistry.hpp"
#include "features/fluid/ParticleBillboardPass.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <span>
#include <string>

namespace lr
{

// Screen-space surface reconstruction for liquid particles. The pass keeps simulation data and
// rendering deliberately separate: callers upload the same ParticleGpuData records used by the
// reference billboard renderer, while this class turns them into a smooth refractive surface.
class ScreenSpaceLiquidPass
{
public:
    struct Config
    {
        std::string cameraBufferResourceName;
        std::string particleBufferResourceName = "fluid_liquid_particles";
        uint32_t    maxParticles               = 65536;

        // Bilateral kernel half-width in pixels. The effective radius is the smaller of this and
        // three times spatialSigma, so widening the blur means raising BOTH: whichever is lower
        // binds. Fusing neighbouring impostors into one surface needs a radius on the order of a
        // particle's on-screen diameter, so the default here is only right for particles of a few
        // pixels; larger impostors stay visibly separate until this is raised to match.
        int   filterRadius      = 5;
        float spatialSigma      = 3.0f;
        // Fraction of the centre sample's view distance, not an absolute length, so the filter
        // preserves a silhouette the same way at two metres and at fifty.
        float depthSigma        = 0.02f;
        float thicknessScale    = 1.0f;
        float refractionStrength = 0.018f;
        float indexOfRefraction = 1.333f;
        float absorptionScale   = 1.35f;
        glm::vec3 liquidColor{0.25f, 0.5f, 0.95f};
    };

    ScreenSpaceLiquidPass(Config config, ResourceRegistry &resources);

    void build(FrameGraph &frameGraph);
    void setParticles(std::span<const ParticleGpuData> particles);
    void setLiquidColor(const glm::vec3 &color);
    void updateParams();

    uint32_t particleCount() const { return m_particleCount; }
    Config       &config() { return m_config; }
    const Config &config() const { return m_config; }

    static constexpr const char *compositeImageName() { return "fluidComposite"; }

private:
    Config            m_config;
    ResourceRegistry *m_resources     = nullptr;
    uint32_t          m_particleCount = 0;
};

namespace screen_space_liquid
{

inline float fresnelF0(float indexOfRefraction)
{
    const float ratio = (1.0f - indexOfRefraction) / (1.0f + indexOfRefraction);
    return ratio * ratio;
}

inline glm::vec3 absorptionCoefficient(const glm::vec3 &color, float scale)
{
    return glm::max(glm::vec3(1.0f) - glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f)), glm::vec3(0.02f)) *
           glm::max(scale, 0.0f);
}

inline uint32_t dispatchSize(uint32_t pixels, uint32_t groupSize = 16)
{
    return (pixels + groupSize - 1) / groupSize;
}

} // namespace screen_space_liquid

} // namespace lr
