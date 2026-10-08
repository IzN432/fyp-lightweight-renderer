#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/passes/geometry/GeometryPass.hpp"

namespace lr
{

struct alignas(16) SpotShadowGpuData
{
    static constexpr uint32_t maxShadows = 16;
    glm::mat4  lightViewProj[maxShadows]{};
    // Per-shadow PCSS inputs, one vec4 per shadow.
    // x = the emitter's radius projected into shadow-map UV at the near plane (0 disables PCSS),
    // y = near plane, z = far plane; w is reserved.
    glm::vec4  pcss[maxShadows]{};
    glm::uvec4 header{}; // x = active shadow count
};

class SpotShadowPass
{
public:
    struct Config
    {
        GeometryPass::Config geometry;
        std::vector<SceneObject *> lightObjects;
        uint32_t             resolution  = 2048;
        std::string          shadowImage = "spotShadowMap";
        std::string          paramsBuffer = "spotShadowParams";
    };

    SpotShadowPass(ResourceRegistry &resources, Config cfg);
    void build(FrameGraph &fg, const GpuMeshLayout &layout) const;
    void setSceneGeometry(SceneDrawList draws) { m_cfg.geometry.draws = std::move(draws); }
    void setLightObjects(std::vector<SceneObject *> lights) { m_cfg.lightObjects = std::move(lights); }

    const std::string &shadowImageName() const { return m_cfg.shadowImage; }
    const std::string &paramsBufferName() const { return m_cfg.paramsBuffer; }

private:
    SpotShadowGpuData shadowData() const;

    ResourceRegistry &m_resources;
    Config            m_cfg;
};

} // namespace lr
