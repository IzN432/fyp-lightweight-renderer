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
        .uniformBuffer(0, fg.buffer(m_cfg.cameraBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(1, fg.image("ibl_env"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledDepth(2, fg.image("gbufferDepth"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledDepth(3, fg.image("overlayDepth"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(4, fg.image("pbr"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(5, fg.image("overlay"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(6, fg.image("overlayPoints"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(7, fg.image("heatmap"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .colorAttachment(fg.image("swapchain"), m_cfg.swapchainFormat)
        .execute([](PassContext &ctx) {
            ctx.cmd().draw(3);
        });
}

} // namespace lr
