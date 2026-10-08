#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/passes/geometry/GeometryPass.hpp"

namespace lr
{

// Shadows for rectangular area lights. Each active face emits within its authored spread angle and
// gets a matching perspective map aimed along the quad's axis from behind it (see
// AreaShadowPass::shadowData for how the eye distance and near plane fall out of the quad's size).
// A two-sided light illuminates two disjoint cones and a fragment lies in exactly one, so the faces
// are independent maps and shading samples only the one facing it.
//
// A one-sided light still gets both layers built and drawn, which keeps the layers contiguous for
// instanced rendering. Nothing has to flag the unused face: LTC already returns zero behind a
// one-sided quad, so whatever visibility its back layer reports is multiplied away.
struct alignas(16) AreaShadowGpuData
{
    static constexpr uint32_t sidesPerLight = 2; // front face, then back face
    // Deliberately small: every layer is a full depth target and an area light costs two of them.
    static constexpr uint32_t maxLights = 4;
    static constexpr uint32_t maxLayers = sidesPerLight * maxLights;
    glm::mat4  lightViewProj[maxLayers]{};
    // Mirrors SpotShadowGpuData::pcss, std140 vec4 stride included, because an area light's shadow
    // is the same perspective PCSS lookup. x = the emitter's radius in shadow-map UV at the near
    // plane, y = near plane, z = far plane; w is reserved.
    glm::vec4  pcss[maxLayers]{};
    glm::uvec4 header{}; // x = active area-light shadows, y = layers per light
};

class AreaShadowPass
{
public:
    struct Config
    {
        GeometryPass::Config       geometry;
        std::vector<SceneObject *> lightObjects;
        uint32_t                   resolution = 1024;
        // How far in front of the quad its shadows still matter; sets the projection's far plane.
        float                      range        = 50.0f;
        std::string                shadowImage  = "areaShadowMap";
        std::string                paramsBuffer = "areaShadowParams";
    };

    AreaShadowPass(ResourceRegistry &resources, Config cfg);
    void build(FrameGraph &fg, const GpuMeshLayout &layout) const;
    void setSceneGeometry(VertexBufferUploadResult vertices, IndexBufferUploadResult indices,
                          std::vector<const TransformComponent *> transforms, std::vector<SceneObject *> objects,
                          std::vector<SkinDrawInfo> skins);
    void setLightObjects(std::vector<SceneObject *> lights) { m_cfg.lightObjects = std::move(lights); }
    Config            &config() { return m_cfg; }
    const Config      &config() const { return m_cfg; }
    const std::string &shadowImageName() const { return m_cfg.shadowImage; }
    const std::string &paramsBufferName() const { return m_cfg.paramsBuffer; }

private:
    AreaShadowGpuData shadowData() const;
    ResourceRegistry &m_resources;
    Config            m_cfg;
};
} // namespace lr
