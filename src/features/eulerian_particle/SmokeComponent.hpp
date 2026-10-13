#pragma once

#include "core/scene/Component.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>

namespace lr::eulerian
{

// Authored smoke settings. The CUDA solver and its Vulkan resources are shared transient runtime
// state; this component is the scene-owned part that can be selected, copied and persisted.
class SmokeComponent final : public Component
{
public:
    struct StepConfig
    {
        float buoyancy           = 0.1f;
        float gravity            = 1.0f;
        float ambientTemperature = 20.0f;
        float cooling            = 1.0f;
        float velocityDamp       = 0.0f;
        float densityDamp        = 0.15f;
        float vorticity          = 2.0f;
        bool  openTop            = false;

        bool operator==(const StepConfig &) const = default;
    };

    struct SourceConfig
    {
        glm::vec3 position{0.0f, 0.45f, 0.0f};
        float     radius      = 0.25f;
        float     density     = 18.0f;
        glm::vec3 velocity{0.0f, 5.0f, 0.0f};
        float     temperature = 400.0f;
        float     heating     = 8.0f;
        float     drift       = 0.0f;

        bool operator==(const SourceConfig &) const = default;
    };

    struct Config
    {
        glm::uvec3 gridSize{64, 96, 64};
        float      simulationScale = 2.0f / 64.0f;
        // Local offset from the owning object's world position. Rotation and scale are ignored,
        // matching FluidComponent's translation-only presentation transform.
        glm::vec3 boundsCentre{0.0f, 1.5f, 0.0f};
        bool      showBounds = true;
        bool      enabled    = true;
        bool      startPaused = false;
        bool      voxelizeObstacles = true;

        int        jacobiIterations = 140;
        StepConfig step{};
        SourceConfig source{};

        glm::vec3 coolColor{0.74f, 0.79f, 0.90f};
        glm::vec3 warmColor{1.00f, 0.58f, 0.26f};
        float     coolTemperature = 20.0f;
        float     warmTemperature = 180.0f;
        float     densityScale    = 5.0f;
        int       sampleCount     = 128;

        bool operator==(const Config &) const = default;
    };

    explicit SmokeComponent(Config config = {});

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

} // namespace lr::eulerian
