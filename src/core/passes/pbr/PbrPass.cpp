#include "PbrPass.hpp"

#include "core/Paths.hpp"
#include "core/upload/CameraUploader.hpp"

namespace lr
{

struct PbrPC
{
    uint32_t pfMips; // prefiltered environment map mip levels
    uint32_t numLights;
};

PbrPass::PbrPass(Config cfg)
    : m_cfg(std::move(cfg))
{
}

void PbrPass::build(FrameGraph &fg) const
{
    const PbrPC pbrPC{
        .pfMips = m_cfg.pfMips,
        .numLights = m_cfg.numLights
    };

    fg.addPass("pbr")
        .type(PassType::Fullscreen)
        .vertShader((paths::shaderDir / "fullscreen.vert.spv").string())
        .fragShader((paths::shaderDir / "pbr.frag.spv").string())
        .pushConstantSize(sizeof(PbrPC), VK_SHADER_STAGE_FRAGMENT_BIT) // numLights as push constant
        .bind({
            {
                .resourceName = m_cfg.cameraBufferResourceName,
                .binding      = 0,
                .type         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName = "ibl_irradiance",
                .binding      = 1,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "ibl_prefiltered",
                .binding      = 2,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "ibl_brdf_lut",
                .binding      = 3,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "gbufferDepth",
                .binding      = 4,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "gbufferAlbedo",
                .binding      = 5,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "gbufferNormal",
                .binding      = 6,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "gbufferMaterial",
                .binding      = 7,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = m_cfg.lightBufferResourceName,
                .binding      = 8,
                .type         = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
            }
        })
        .writes({
            {.name = "pbr",   .format = VK_FORMAT_R16G16B16A16_SFLOAT},
        })
        .execute([pbrPC](CommandBuffer &cmd, VkPipelineLayout layout) {
            cmd.pushConstants(layout, VK_SHADER_STAGE_FRAGMENT_BIT, pbrPC);
            cmd.draw(3);
        });
}

}  // namespace lr
