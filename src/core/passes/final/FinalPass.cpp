#include "FinalPass.hpp"

#include "core/Paths.hpp"
#include "core/upload/CameraUploader.hpp"

namespace lr
{

FinalPass::FinalPass(Config cfg) : m_cfg(std::move(cfg)) {}

void FinalPass::build(FrameGraph &fg) const
{
    fg.addPass("final")
        .type(PassType::Fullscreen)
        .vertShader((paths::shaderDir / "fullscreen.vert.spv").string())
        .fragShader((paths::shaderDir / "final.frag.spv").string())
        .bind({{
                   .binding = 0,
                   .type    = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                   .stages  = VK_SHADER_STAGE_FRAGMENT_BIT,
                   .buffer  = fg.buffer(m_cfg.cameraBufferResourceName),
               },
               {
                   .binding     = 1,
                   .type        = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   .stages      = VK_SHADER_STAGE_FRAGMENT_BIT,
                   .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   .image       = fg.image("ibl_env"),
               },
               {
                   .binding     = 2,
                   .type        = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   .stages      = VK_SHADER_STAGE_FRAGMENT_BIT,
                   .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                   .image       = fg.image("gbufferDepth"),
               },
               {
                   .binding     = 3,
                   .type        = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   .stages      = VK_SHADER_STAGE_FRAGMENT_BIT,
                   .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                   .image       = fg.image("overlayDepth"),
               },
               {
                   .binding     = 4,
                   .type        = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   .stages      = VK_SHADER_STAGE_FRAGMENT_BIT,
                   .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   .image       = fg.image("pbr"),
               },
               {
                   .binding     = 5,
                   .type        = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   .stages      = VK_SHADER_STAGE_FRAGMENT_BIT,
                   .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   .image       = fg.image("overlay"),
               },
               {
                   .binding     = 6,
                   .type        = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   .stages      = VK_SHADER_STAGE_FRAGMENT_BIT,
                   .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   .image       = fg.image("overlayPoints"),
               },
               {
                   .binding     = 7,
                   .type        = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   .stages      = VK_SHADER_STAGE_FRAGMENT_BIT,
                   .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   .image       = fg.image("heatmap"),
               }})
        .writes({
            {.format = m_cfg.swapchainFormat, .image = fg.image("swapchain")},
        })
        .execute([](CommandBuffer &cmd, VkPipelineLayout) {
            cmd.draw(3);
        });
}

} // namespace lr
