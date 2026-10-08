#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/passes/geometry/GeometryPass.hpp"

namespace lr
{
struct alignas(16) CascadedShadowGpuData
{
    static constexpr uint32_t cascadeCount = 4;
    // CSM is reserved for the first directional light (the scene's sun). Additional directional
    // lights still shade normally but do not multiply the four-layer shadow allocation.
    static constexpr uint32_t maxLights = 1;
    static constexpr uint32_t maxLayers = cascadeCount * maxLights;
    glm::mat4 lightViewProj[maxLayers]{};
    glm::vec4 splitDepths{};
    glm::uvec4 header{}; // x = active directional lights, y = cascades per light
};

class CascadedShadowPass
{
public:
    struct Config
    {
        GeometryPass::Config geometry;
        std::vector<SceneObject *> lightObjects;
        SceneObject *camera = nullptr;
        uint32_t resolution = 2048;
        float shadowDistance = 200.0f;
        float splitLambda = 0.75f;
        float depthPadding = 50.0f;
        std::string shadowImage = "cascadedShadowMap";
        std::string paramsBuffer = "cascadedShadowParams";
    };

    CascadedShadowPass(ResourceRegistry &resources, Config cfg);
    void build(FrameGraph &fg, const GpuMeshLayout &layout) const;
    void setSceneGeometry(VertexBufferUploadResult vertices, IndexBufferUploadResult indices,
                          std::vector<const TransformComponent *> transforms, std::vector<SceneObject *> objects,
                          std::vector<SkinDrawInfo> skins);
    void setLightObjects(std::vector<SceneObject *> lights) { m_cfg.lightObjects = std::move(lights); }
    void setViewportExtent(VkExtent2D extent);
    Config &config() { return m_cfg; }
    const Config &config() const { return m_cfg; }
    const std::string &shadowImageName() const { return m_cfg.shadowImage; }
    const std::string &paramsBufferName() const { return m_cfg.paramsBuffer; }

private:
    CascadedShadowGpuData shadowData() const;
    ResourceRegistry &m_resources;
    Config m_cfg;
    float m_aspect = 16.0f / 9.0f;
};
} // namespace lr
