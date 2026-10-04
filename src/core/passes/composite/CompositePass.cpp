#include "CompositePass.hpp"

#include "core/Paths.hpp"

namespace lr
{

CompositePass::CompositePass(Config cfg) : m_cfg(std::move(cfg)) {}

void CompositePass::build(FrameGraph &fg) const
{
    fg.addPass(m_cfg.passName)
        .type(PassType::Fullscreen)
        .vertShader((paths::shaderDir / "fullscreen.vert.spv").string())
        .fragShader((paths::shaderDir / "composite.frag.spv").string())
        .uniformBuffer(0, fg.buffer(m_cfg.cameraBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(1, fg.image("ibl_env"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledDepth(2, fg.image("gbufferDepth"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(3, fg.image(m_cfg.inputImage), VK_SHADER_STAGE_FRAGMENT_BIT)
        .colorAttachment(fg.image(m_cfg.outputImage), m_cfg.outputFormat)
        .execute([](PassContext &ctx) {
            ctx.cmd().draw(3);
        });
}

} // namespace lr
