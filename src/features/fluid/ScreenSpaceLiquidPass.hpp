#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/framegraph/ResourceRegistry.hpp"
#include "features/fluid/ParticleBillboardPass.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <bit>

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
        // Whether this pass allocates the particle buffer. False means another component already
        // registered it and this pass only reads it -- which is how the GPU solver feeds the
        // renderer: it writes the records in a compute pass, so the buffer must be its device-local
        // one rather than a dynamic buffer this pass would own and upload to. setParticles() is
        // then unavailable; the producer sets the count with setParticleCount() instead.
        bool ownsParticleBuffer = true;

        // Bilateral kernel half-width in WORLD units. The shader projects this to a per-pixel
        // radius at the sampled depth, so the kernel covers the same amount of liquid at any
        // camera distance. Fusing neighbouring impostors into one surface takes a radius on the
        // order of a particle diameter, so this tracks the particle size rather than a fixed
        // pixel count; smoothingRadiusFromParticle() derives a sensible value.
        float smoothingRadius = 0.14f;
        // Hard ceiling on the projected radius, in pixels. This bounds the per-pixel cost when
        // the liquid comes close to the camera; it is a performance guard, not a tuning knob, so
        // it sits far above the radius any reasonable shot needs.
        int maxFilterRadius = 96;
        // Fraction of the centre sample's view distance, not an absolute length, so the filter
        // preserves a silhouette the same way at two metres and at fifty.
        float     depthSigma         = 0.02f;
        float     thicknessScale     = 1.0f;
        float     refractionStrength = 0.018f;
        float     indexOfRefraction  = 1.333f;
        float     absorptionScale    = 1.35f;
        glm::vec3 liquidColor{0.25f, 0.5f, 0.95f};
        // Presentation-only uniform scale for solver positions and particle radii.
        float renderScale = 1.0f;
    };

    ScreenSpaceLiquidPass(Config config, ResourceRegistry &resources);

    void build(FrameGraph &frameGraph);
    void setParticles(std::span<const ParticleGpuData> particles);
    // How many records of an externally written buffer to draw.
    void setParticleCount(uint32_t count);
    void setLiquidColor(const glm::vec3 &color);
    // Kernel width in world units; the shader projects it per pixel. Kept in sync with the
    // simulation's particle size, so it changes whenever spacing or render scale do.
    void setSmoothingRadius(float radius);
    void setRenderScale(float scale) { m_config.renderScale = std::max(scale, 0.0f); }
    float renderScale() const { return m_config.renderScale; }
    void updateParams();

    uint32_t      particleCount() const { return m_particleCount; }
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

// CPU reference for fluid_composite.frag. Absorption remains coloured, while one scalar extinction
// weight controls the visible bulk tint so its hue cannot be inverted channel by channel.
inline glm::vec3 compositeTransmission(const glm::vec3 &background, const glm::vec3 &color,
                                       float absorptionScale, float thickness)
{
    const glm::vec3 tint = glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f));
    const glm::vec3 transmittance =
        glm::exp(-absorptionCoefficient(tint, absorptionScale) * std::max(thickness, 0.0f));
    const float tintWeight = 1.0f - (transmittance.r + transmittance.g + transmittance.b) / 3.0f;
    return glm::mix(background * transmittance, tint, tintWeight);
}

// Optical path through the rasterized particle volume. Invalid or crossed surfaces yield no
// thickness; the scale remains an artistic density control rather than compensating for particle
// overlap as the former additive-chord approximation required.
inline float thicknessFromDepths(float frontDepth, float backDepth, float scale = 1.0f)
{
    return std::max(backDepth - frontDepth, 0.0f) * std::max(scale, 0.0f);
}

// The GPU divides the front-to-back span into this many slices and marks only slices intersected
// by a particle chord. This bounded representation preserves empty gaps without storing an
// unbounded per-pixel fragment list.
inline constexpr uint32_t occupancySliceCount = 32;

inline float occupiedThickness(float frontDepth, float backDepth, uint32_t occupancyMask, float scale = 1.0f)
{
    const float span = std::max(backDepth - frontDepth, 0.0f);
    return span * (static_cast<float>(std::popcount(occupancyMask)) / static_cast<float>(occupancySliceCount)) *
           std::max(scale, 0.0f);
}

inline float firstIntervalThickness(float frontDepth, float backDepth, uint32_t occupancyMask,
                                    float scale = 1.0f)
{
    uint32_t occupiedSlices = 0;
    while (occupiedSlices < occupancySliceCount && (occupancyMask & (uint32_t{1} << occupiedSlices)) != 0)
    {
        ++occupiedSlices;
    }
    const float span = std::max(backDepth - frontDepth, 0.0f);
    return span * (static_cast<float>(occupiedSlices) / static_cast<float>(occupancySliceCount)) *
           std::max(scale, 0.0f);
}

// Neighbouring impostors only merge into a continuous surface once the kernel spans roughly a
// particle diameter; below that each sphere keeps its own depth profile and shades separately.
inline float smoothingRadiusFromParticle(float particleRadius) { return std::max(particleRadius, 0.0f) * 2.0f; }

inline uint32_t dispatchSize(uint32_t pixels, uint32_t groupSize = 16) { return (pixels + groupSize - 1) / groupSize; }

} // namespace screen_space_liquid

} // namespace lr
