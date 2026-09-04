#include "PbrPass.hpp"

#include "core/Paths.hpp"
#include "core/passes/pbr/LtcMatrix.hpp"
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
}

void PbrPass::build(FrameGraph &fg) const
{
    const PbrPC pbrPC{.pfMips = m_cfg.pfMips, .numLights = m_cfg.numLights};

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
        .sampledDepth(6, fg.image("gbufferDepth"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(7, fg.image("gbufferAlbedo"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(8, fg.image("gbufferNormal"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(9, fg.image("gbufferMaterial"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(10, fg.image("gbufferEmissive"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(11, fg.buffer(m_cfg.lightBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(12, fg.image("hbao_ao"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .colorAttachment(fg.image("pbr"), VK_FORMAT_R16G16B16A16_SFLOAT)
        .execute([pbrPC](CommandBuffer &cmd, VkPipelineLayout layout) {
            cmd.pushConstants(layout, VK_SHADER_STAGE_FRAGMENT_BIT, pbrPC);
            cmd.draw(3);
        });
}

} // namespace lr
