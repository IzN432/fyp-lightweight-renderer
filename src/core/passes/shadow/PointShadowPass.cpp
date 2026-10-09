#include "PointShadowPass.hpp"

#include "core/Paths.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Scene.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>

namespace lr
{
namespace
{
struct ShadowPC { glm::mat4 model; uint32_t primitiveIdOffset; uint32_t paletteOffset; uint32_t skinEnabled; };
constexpr glm::vec3 directions[6] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
constexpr glm::vec3 ups[6] = {{0,-1,0},{0,-1,0},{0,0,1},{0,0,-1},{0,-1,0},{0,-1,0}};
}

PointShadowPass::PointShadowPass(ResourceRegistry &resources, Config cfg)
    : m_resources(resources), m_cfg(std::move(cfg))
{
    if (m_cfg.resolution == 0) throw std::invalid_argument("PointShadowPass: invalid configuration");
    resources.registerPersistentImageArray(m_cfg.shadowImage, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        {m_cfg.resolution, m_cfg.resolution}, PointShadowGpuData::maxLayers, VK_IMAGE_ASPECT_DEPTH_BIT);
    resources.registerDynamicBuffer(m_cfg.paramsBuffer, sizeof(PointShadowGpuData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    const PointShadowGpuData disabled{};
    resources.updateBuffer(m_cfg.paramsBuffer, &disabled, sizeof(disabled));
}

PointShadowGpuData PointShadowPass::shadowData() const
{
    PointShadowGpuData result{};
    result.header.y = PointShadowGpuData::facesPerLight;
    for (SceneObject *object : m_cfg.lightObjects)
    {
        if (!object || !object->scene().contains(object->id()) || !object->hasComponent<Light>()) continue;
        const auto *point = std::get_if<PointLight>(&object->getComponent<Light>().light);
        if (!point) continue;
        const glm::vec3 position = object->getComponent<TransformComponent>().transform().position();
        glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, point->shadowNearPlane, point->range);
        projection[1][1] *= -1.0f;
        glm::mat4 clip(1.0f); clip[2][2] = 0.5f; clip[3][2] = 0.5f;
        const uint32_t shadowIndex = result.header.x++;
        for (uint32_t face = 0; face < PointShadowGpuData::facesPerLight; ++face)
        {
            const uint32_t layer = shadowIndex * PointShadowGpuData::facesPerLight + face;
            result.lightViewProj[layer] = clip * projection * glm::lookAt(position, position + directions[face], ups[face]);
            result.pcss[layer] = glm::vec4(point->sourceRadius / (2.0f * point->shadowNearPlane),
                                           point->shadowNearPlane, point->range, 0.0f);
        }
        if (result.header.x == PointShadowGpuData::maxShadows) break;
    }
    return result;
}

void PointShadowPass::build(FrameGraph &fg, const GpuMeshLayout &layout) const
{
    auto pass = fg.addPass("pointShadow").type(PassType::Geometry).vertexLayout(layout);
    for (const auto &[binding, name] : m_cfg.geometry.vertexBufferResourceNames) pass.vertexBuffer(binding, fg.buffer(name));
    pass.indexBuffer(fg.buffer(m_cfg.geometry.indexBufferResourceName))
        .vertShader((paths::shaderDir / "point_shadow.vert.spv").string())
        .fragShader((paths::shaderDir / "spot_shadow.frag.spv").string())
        .cull(VK_CULL_MODE_BACK_BIT).depthBias(1.25f, 1.75f)
        .renderingLayers(PointShadowGpuData::maxLayers)
        .pushConstantSize(sizeof(ShadowPC), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .uniformBuffer(0, fg.buffer(m_cfg.paramsBuffer), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledImageArray(1, fg.image(m_cfg.geometry.diffuseTextureArrayResourceName), m_cfg.geometry.materialCount, VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(5, fg.buffer(m_cfg.geometry.faceGroupBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(6, fg.buffer(m_cfg.geometry.materialBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(7, fg.buffer(m_cfg.geometry.skinInfluenceEntriesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(8, fg.buffer(m_cfg.geometry.skinInfluenceOffsetsBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(9, fg.buffer(m_cfg.geometry.skinPositionIndicesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(10, fg.buffer(m_cfg.geometry.skinJointMatricesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .depthAttachment(fg.image(m_cfg.shadowImage), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR,
                         {.depthStencil = {1.0f, 0}}, ExtentSpec::absolute(m_cfg.resolution, m_cfg.resolution))
        .execute([this](PassContext &ctx) {
            const PointShadowGpuData data = shadowData();
            m_resources.updateBuffer(m_cfg.paramsBuffer, &data, sizeof(data));
            for (size_t i = 0; i < m_cfg.geometry.draws.size() && data.header.x != 0; ++i)
            {
                if (!m_cfg.geometry.draws.isLive(i)) continue;
                const SceneDraw draw = m_cfg.geometry.draws.at(i, m_skinningEnabled);
                const ShadowPC pc{draw.model, draw.primitiveIdOffset, draw.paletteOffset, draw.skinEnabled};
                ctx.cmd().pushConstants(ctx.pipelineLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
                ctx.cmd().drawIndexed(draw.indexCount, data.header.x * PointShadowGpuData::facesPerLight,
                                      draw.firstIndex, draw.vertexOffset, 0);
            }
        });
}
}
