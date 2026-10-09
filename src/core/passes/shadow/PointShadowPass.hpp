#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/passes/geometry/GeometryPass.hpp"

namespace lr
{
struct alignas(16) PointShadowGpuData
{
    static constexpr uint32_t maxShadows = 4;
    static constexpr uint32_t facesPerLight = 6;
    static constexpr uint32_t maxLayers = maxShadows * facesPerLight;
    glm::mat4 lightViewProj[maxLayers]{};
    glm::vec4 pcss[maxLayers]{};
    glm::uvec4 header{}; // x = active lights, y = faces per light
};

class PointShadowPass
{
public:
    struct Config
    {
        GeometryPass::Config geometry;
        std::vector<SceneObject *> lightObjects;
        uint32_t resolution = 1024;
        std::string shadowImage = "pointShadowMap";
        std::string paramsBuffer = "pointShadowParams";
    };

    PointShadowPass(ResourceRegistry &resources, Config cfg);
    void build(FrameGraph &fg, const GpuMeshLayout &layout) const;
    void setSceneGeometry(SceneDrawList draws) { m_cfg.geometry.draws = std::move(draws); }
    void setLightObjects(std::vector<SceneObject *> lights) { m_cfg.lightObjects = std::move(lights); }
    void setSkinningEnabled(bool enabled) { m_skinningEnabled = enabled; }
    const std::string &shadowImageName() const { return m_cfg.shadowImage; }
    const std::string &paramsBufferName() const { return m_cfg.paramsBuffer; }

private:
    PointShadowGpuData shadowData() const;
    ResourceRegistry &m_resources;
    Config m_cfg;
    bool m_skinningEnabled = true;
};
}
