#include "PbrPass.hpp"

#include "core/upload/CameraUploader.hpp"

namespace lr
{

PbrPass::PbrPass(Config cfg)
    : m_cfg(std::move(cfg))
{
}

void PbrPass::build(FrameGraph &fg) const
{
    const auto &shaderDir = m_cfg.shaderDir;

    fg.addPass("pbr")
        .type(PassType::Fullscreen)
        .vertShader((shaderDir / "fullscreen.vert.spv").string())
        .fragShader((shaderDir / "pbr.frag.spv").string())
        .bind({
            {
                .resourceName = CameraUploader::kBufferName,
                .binding      = 0,
                .type         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName = "ibl_irradiance",
                .binding      = 1,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName = "ibl_prefiltered",
                .binding      = 2,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName = "gbufferDepth",
                .binding      = 3,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "gbufferAlbedo",
                .binding      = 4,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "gbufferNormal",
                .binding      = 5,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "gbufferMaterial",
                .binding      = 6,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
        })
        .writes({
            {.name = "pbr",   .format = VK_FORMAT_R16G16B16A16_SFLOAT},
        })
        .execute([](CommandBuffer &cmd, VkPipelineLayout) {
            cmd.draw(3);
        });
}

}  // namespace lr
