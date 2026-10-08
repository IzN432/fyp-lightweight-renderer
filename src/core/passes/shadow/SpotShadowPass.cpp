#include "SpotShadowPass.hpp"

#include "core/Paths.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>
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
    uint32_t  primitiveIdOffset;
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
            // PCSS works in shadow-map UV, so express the emitter's radius there. The near plane
            // spans 2 * tan(halfFov) * near world units across the full [0, 1] UV range, and the
            // shadow projection's half-FOV is the cone angle widened above, not the cone angle.
            const float tanHalfShadowFov = std::tan(glm::radians(shadowFov) * 0.5f);
            // The near/far travel with the matrix rather than being re-read from the light, so the
            // shader's depth linearisation cannot drift from the projection it has to invert.
            result.pcss[shadowIndex] =
                glm::vec4(spot->sourceRadius / (2.0f * tanHalfShadowFov * spot->shadowNearPlane),
                          spot->shadowNearPlane, spot->range, 0.0f);
            if (result.header.x == SpotShadowGpuData::maxShadows)
                break;
        }
    }
    return result;
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
        .pushConstantSize(sizeof(ShadowPC), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .uniformBuffer(0, fg.buffer(m_cfg.paramsBuffer), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledImageArray(1, fg.image(m_cfg.geometry.diffuseTextureArrayResourceName), m_cfg.geometry.materialCount,
                           VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(5, fg.buffer(m_cfg.geometry.faceGroupBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(6, fg.buffer(m_cfg.geometry.materialBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
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
            const SceneDrawList &draws = m_cfg.geometry.draws;
            for (size_t i = 0; i < draws.size(); ++i)
            {
                if (!draws.isLive(i))
                    continue;
                const SceneDraw draw = draws.at(i, m_skinningEnabled);
                const ShadowPC  pc{.model             = draw.model,
                                   .primitiveIdOffset = draw.primitiveIdOffset,
                                   .paletteOffset     = draw.paletteOffset,
                                   .skinEnabled       = draw.skinEnabled};
                ctx.cmd().pushConstants(ctx.pipelineLayout(),
                                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
                ctx.cmd().drawIndexed(draw.indexCount, data.header.x, draw.firstIndex, draw.vertexOffset, 0);
            }
        });
}

} // namespace lr
