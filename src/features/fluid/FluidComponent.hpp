#pragma once

#include "core/scene/Component.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace lr
{

// Authored fluid settings. The GPU solver remains transient application state; this component is
// the part that belongs to a scene object and is persisted in an .lrscene file.
class FluidComponent final : public Component
{
public:
    struct CuboidSource
    {
        glm::uvec3 dimensions{1, 1, 1};
        glm::vec3  origin{0.0f};

        bool operator==(const CuboidSource &) const = default;
    };

    struct Config
    {
        std::vector<CuboidSource> sources{{{50, 56, 50}, {-14.5f, 0.5f, -14.5f}}};
        uint32_t                  maxParticles = 140000;

        float particleSpacing      = 0.370f;
        float smoothingRadiusScale = 2.0f;
        float particleRenderScale  = 0.9f;
        int   solverIterations     = 8;
        float artificialPressureK  = 0.02f;
        float vorticityEpsilon     = 0.0f;
        float viscosity            = 0.03f;
        float boundsFriction       = 0.98f;
        float maxDisplacementScale = 0.5f;

        glm::vec3 boundsMin{-15.0f, 0.0f, -15.0f};
        glm::vec3 boundsMax{15.0f, 35.0f, 15.0f};
        bool      showBounds       = true;
        float     fixedDeltaTime   = 0.016f;
        int       maxStepsPerFrame = 2;
        bool      startPaused      = false;
        glm::vec3 particleColor{0.25f, 0.5f, 0.95f};
        // Solver coordinates are normalized for numerical stability. This presentation scale and
        // the owning object's position place them in the authored scene; rotation and scale do not.
        float renderScale = 1.0f / 35.0f;

        float smoothingRadius() const;
        float particleRadius() const;

        bool operator==(const Config &) const = default;
    };

    explicit FluidComponent(Config config = {});

    const Config &config() const { return m_config; }
    uint64_t      revision() const { return m_revision; }
    void          setConfig(Config config);

    std::unique_ptr<ComponentValues> copyValues() const override;
    void                             pasteValues(const ComponentValues &values) override;
    ComponentValuesAdder             valuesAdder() const override;
    bool                             allowsRemoval() const override { return true; }
    void                             onGUIImpl() override;

private:
    Config   m_config;
    uint64_t m_revision = 0;
};

} // namespace lr
