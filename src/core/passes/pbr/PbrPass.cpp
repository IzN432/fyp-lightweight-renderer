#include "PbrPass.hpp"

#include "core/Paths.hpp"
#include "core/passes/pbr/LtcMatrix.hpp"
#include "core/passes/shadow/SpotShadowPass.hpp"
#include "core/passes/shadow/CascadedShadowPass.hpp"
#include "core/passes/shadow/AreaShadowPass.hpp"
#include "core/passes/shadow/PointShadowPass.hpp"
#include "core/upload/CameraUploader.hpp"

namespace lr
{

namespace
{
constexpr uint32_t kLtcSize = 64;
}

struct PbrPC
{
    uint32_t pfMips; // prefiltered environment map mip levels
    uint32_t numLights;
};

PbrPass::PbrPass(Config cfg) : m_cfg(std::move(cfg)) {}

void PbrPass::uploadResources(ResourceRegistry &resources) const
{
    // LTC lookup tables for area light shading (Heitz et al.) — LTC1 holds the
    // inverse transform M^-1, LTC2 holds GGX norm/fresnel/horizon-clip terms,
    // both indexed by (roughness, cosTheta) over a 64x64 grid.
    resources.uploadImage("ltc1", LTC1, kLtcSize, kLtcSize, VK_FORMAT_R32G32B32A32_SFLOAT);
    resources.uploadImage("ltc2", LTC2, kLtcSize, kLtcSize, VK_FORMAT_R32G32B32A32_SFLOAT);
    if (!resources.hasImage(m_cfg.shadowImageResourceName))
    {
        resources.registerPersistentImageArray(m_cfg.shadowImageResourceName, VK_FORMAT_D32_SFLOAT,
                                               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                                   VK_IMAGE_USAGE_SAMPLED_BIT,
                                               {1, 1}, SpotShadowGpuData::maxShadows,
                                               VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    if (!resources.hasBuffer(m_cfg.shadowParamsBufferResourceName))
    {
        resources.registerDynamicBuffer(m_cfg.shadowParamsBufferResourceName, sizeof(SpotShadowGpuData),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        const SpotShadowGpuData disabled{};
        resources.updateBuffer(m_cfg.shadowParamsBufferResourceName, &disabled, sizeof(disabled));
    }
    if (!resources.hasImage(m_cfg.cascadedShadowImageResourceName))
    {
        resources.registerPersistentImageArray(m_cfg.cascadedShadowImageResourceName, VK_FORMAT_D32_SFLOAT,
                                               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                                   VK_IMAGE_USAGE_SAMPLED_BIT,
                                               {1, 1}, CascadedShadowGpuData::maxLayers,
                                               VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    if (!resources.hasBuffer(m_cfg.cascadedShadowParamsBufferResourceName))
    {
        resources.registerDynamicBuffer(m_cfg.cascadedShadowParamsBufferResourceName,
                                        sizeof(CascadedShadowGpuData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        const CascadedShadowGpuData disabled{};
        resources.updateBuffer(m_cfg.cascadedShadowParamsBufferResourceName, &disabled, sizeof(disabled));
    }
    if (!resources.hasImage(m_cfg.areaShadowImageResourceName))
    {
        resources.registerPersistentImageArray(m_cfg.areaShadowImageResourceName, VK_FORMAT_D32_SFLOAT,
                                               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                                   VK_IMAGE_USAGE_SAMPLED_BIT,
                                               {1, 1}, AreaShadowGpuData::maxLayers,
                                               VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    if (!resources.hasBuffer(m_cfg.areaShadowParamsBufferResourceName))
    {
        resources.registerDynamicBuffer(m_cfg.areaShadowParamsBufferResourceName, sizeof(AreaShadowGpuData),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        const AreaShadowGpuData disabled{};
        resources.updateBuffer(m_cfg.areaShadowParamsBufferResourceName, &disabled, sizeof(disabled));
    }
    if (!resources.hasImage(m_cfg.pointShadowImageResourceName))
        resources.registerPersistentImageArray(m_cfg.pointShadowImageResourceName, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            {1, 1}, PointShadowGpuData::maxLayers, VK_IMAGE_ASPECT_DEPTH_BIT);
    if (!resources.hasBuffer(m_cfg.pointShadowParamsBufferResourceName))
    {
        resources.registerDynamicBuffer(m_cfg.pointShadowParamsBufferResourceName, sizeof(PointShadowGpuData),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        const PointShadowGpuData disabled{};
        resources.updateBuffer(m_cfg.pointShadowParamsBufferResourceName, &disabled, sizeof(disabled));
    }
}

void PbrPass::build(FrameGraph &fg) const
{
    fg.addPass("pbr")
        .type(PassType::Fullscreen)
        .vertShader((paths::shaderDir / "fullscreen.vert.spv").string())
        .fragShader((paths::shaderDir / "pbr.frag.spv").string())
        .pushConstantSize(sizeof(PbrPC), VK_SHADER_STAGE_FRAGMENT_BIT)
        .uniformBuffer(0, fg.buffer(m_cfg.cameraBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(1, fg.image("ibl_irradiance"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(2, fg.image("ibl_prefiltered"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(3, fg.image("ibl_brdf_lut"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(4, fg.image("ltc1"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(5, fg.image("ltc2"), VK_SHADER_STAGE_FRAGMENT_BIT)
        // PBR shades each G-buffer coverage sample independently. Reading the unresolved attachments
        // avoids lighting material values that were incorrectly averaged across a triangle edge.
        .sampledMultisampleImage(6, fg.image("gbufferDepth"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledMultisampleImage(7, fg.image("gbufferAlbedo"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledMultisampleImage(8, fg.image("gbufferNormal"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledMultisampleImage(9, fg.image("gbufferMaterial"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledMultisampleImage(10, fg.image("gbufferEmissive"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(11, fg.buffer(m_cfg.lightBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(12, fg.image("hbao_ao"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledDepth(13, fg.image(m_cfg.shadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::shadowComparison())
        .uniformBuffer(14, fg.buffer(m_cfg.shadowParamsBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledDepth(15, fg.image(m_cfg.cascadedShadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::shadowComparison())
        .uniformBuffer(16, fg.buffer(m_cfg.cascadedShadowParamsBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        // Both shadow maps a second time, uncompared, for the PCSS blocker search.
        .sampledDepth(17, fg.image(m_cfg.shadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::depthFetch())
        .sampledDepth(18, fg.image(m_cfg.cascadedShadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::depthFetch())
        .sampledDepth(19, fg.image(m_cfg.areaShadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::shadowComparison())
        .uniformBuffer(20, fg.buffer(m_cfg.areaShadowParamsBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledDepth(21, fg.image(m_cfg.areaShadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::depthFetch())
        .sampledDepth(22, fg.image(m_cfg.pointShadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::shadowComparison())
        .uniformBuffer(23, fg.buffer(m_cfg.pointShadowParamsBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledDepth(24, fg.image(m_cfg.pointShadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::depthFetch())
        .colorAttachment(fg.image("pbr"), VK_FORMAT_R16G16B16A16_SFLOAT)
        // Reads m_cfg when the pass runs, so setNumLights() takes effect without rebuilding the graph.
        .execute([this](PassContext &ctx) {
            const PbrPC pbrPC{.pfMips = m_cfg.pfMips, .numLights = m_cfg.numLights};
            ctx.cmd().pushConstants(ctx.pipelineLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, pbrPC);
            ctx.cmd().draw(3);
        });
}

} // namespace lr
