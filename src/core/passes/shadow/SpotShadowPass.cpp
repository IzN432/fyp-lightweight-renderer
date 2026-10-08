#include "SpotShadowPass.hpp"

#include "core/Paths.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <stdexcept>

namespace lr
{
namespace
{
constexpr float kShadowFovMarginDegrees = 5.0f;
constexpr float kMaxShadowFovDegrees    = 179.0f;

struct ShadowPC
{
    glm::mat4 model;
    uint32_t  paletteOffset;
    uint32_t  skinEnabled;
};
}

SpotShadowPass::SpotShadowPass(ResourceRegistry &resources, Config cfg)
    : m_resources(resources), m_cfg(std::move(cfg))
{
    if (m_cfg.resolution == 0)
    {
        throw std::invalid_argument("SpotShadowPass: invalid configuration");
    }
    resources.registerPersistentImageArray(m_cfg.shadowImage, VK_FORMAT_D32_SFLOAT,
                                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                           {m_cfg.resolution, m_cfg.resolution}, SpotShadowGpuData::maxShadows,
                                           VK_IMAGE_ASPECT_DEPTH_BIT);
    resources.registerDynamicBuffer(m_cfg.paramsBuffer, sizeof(SpotShadowGpuData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    const SpotShadowGpuData disabled{};
    resources.updateBuffer(m_cfg.paramsBuffer, &disabled, sizeof(disabled));
}

SpotShadowGpuData SpotShadowPass::shadowData() const
{
    SpotShadowGpuData result{};
    for (SceneObject *object : m_cfg.lightObjects)
    {
        if (!object || !object->scene().contains(object->id()) || !object->hasComponent<Light>())
            continue;
        const auto *spot = std::get_if<SpotLight>(&object->getComponent<Light>().light);
        if (spot)
        {
            const Transform &transform = object->getComponent<TransformComponent>().transform();
            const glm::mat4 view = glm::lookAt(transform.position(), transform.position() + transform.forward(),
                                               transform.up());
            const float shadowFov = std::min(spot->outerConeAngleDegrees * 2.0f + kShadowFovMarginDegrees,
                                             kMaxShadowFovDegrees);
            glm::mat4 projection = glm::perspective(glm::radians(shadowFov), 1.0f,
                                                    spot->shadowNearPlane, spot->range);
            projection[1][1] *= -1.0f;
            glm::mat4 clip(1.0f);
            clip[2][2] = 0.5f;
            clip[3][2] = 0.5f;
            const uint32_t shadowIndex = result.header.x++;
            result.lightViewProj[shadowIndex] = clip * projection * view;
            if (result.header.x == SpotShadowGpuData::maxShadows)
                break;
        }
    }
    return result;
}

void SpotShadowPass::setSceneGeometry(VertexBufferUploadResult vertices, IndexBufferUploadResult indices,
                                      std::vector<const TransformComponent *> transforms,
                                      std::vector<SceneObject *> objects, std::vector<SkinDrawInfo> skins)
{
    m_cfg.geometry.vertexBufferUploadResult = std::move(vertices);
    m_cfg.geometry.indexBufferUploadResult  = std::move(indices);
    m_cfg.geometry.meshTransforms           = std::move(transforms);
    m_cfg.geometry.meshObjects              = std::move(objects);
    m_cfg.geometry.skinDrawInfos            = std::move(skins);
}

void SpotShadowPass::build(FrameGraph &fg, const GpuMeshLayout &layout) const
{
    auto pass = fg.addPass("spotShadow").type(PassType::Geometry).vertexLayout(layout);
    for (const auto &[binding, name] : m_cfg.geometry.vertexBufferResourceNames)
        pass.vertexBuffer(binding, fg.buffer(name));

    pass.indexBuffer(fg.buffer(m_cfg.geometry.indexBufferResourceName))
        .vertShader((paths::shaderDir / "spot_shadow.vert.spv").string())
        .fragShader((paths::shaderDir / "spot_shadow.frag.spv").string())
        .cull(VK_CULL_MODE_BACK_BIT)
        .depthBias(1.25f, 1.75f)
        .renderingLayers(SpotShadowGpuData::maxShadows)
        .pushConstantSize(sizeof(ShadowPC), VK_SHADER_STAGE_VERTEX_BIT)
        .uniformBuffer(0, fg.buffer(m_cfg.paramsBuffer), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(7, fg.buffer(m_cfg.geometry.skinInfluenceEntriesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(8, fg.buffer(m_cfg.geometry.skinInfluenceOffsetsBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(9, fg.buffer(m_cfg.geometry.skinPositionIndicesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(10, fg.buffer(m_cfg.geometry.skinJointMatricesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .depthAttachment(fg.image(m_cfg.shadowImage), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR,
                         {.depthStencil = {1.0f, 0}}, ExtentSpec::absolute(m_cfg.resolution, m_cfg.resolution))
        .execute([this](PassContext &ctx) {
            const SpotShadowGpuData data = shadowData();
            m_resources.updateBuffer(m_cfg.paramsBuffer, &data, sizeof(data));
            if (data.header.x == 0)
                return;
            const auto &g = m_cfg.geometry;
            for (size_t i = 0; i < g.vertexBufferUploadResult.singleMeshResults.size(); ++i)
            {
                if (!g.meshObjects[i]->scene().contains(g.meshObjects[i]->id()))
                    continue;
                const auto &mesh  = g.vertexBufferUploadResult.singleMeshResults[i];
                const auto &range = g.indexBufferUploadResult.singleMeshResults[i];
                const auto *transform = g.meshTransforms[i];
                const auto &skin = g.skinDrawInfos[i];
                const ShadowPC pc{transform ? transform->worldMatrix() : glm::mat4(1.0f), skin.paletteOffset,
                                  skin.skinEnabled ? 1u : 0u};
                ctx.cmd().pushConstants(ctx.pipelineLayout(), VK_SHADER_STAGE_VERTEX_BIT, pc);
                ctx.cmd().drawIndexed(range.indexCount, data.header.x, range.firstIndex, mesh.vertexOffset, 0);
            }
        });
}

} // namespace lr
