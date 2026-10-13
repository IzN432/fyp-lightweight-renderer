#include "SmokeRenderPass.hpp"

#include "core/Paths.hpp"

#include <glm/vec4.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lr::eulerian
{
namespace
{

struct alignas(16) SmokeRenderPC
{
    glm::vec4 boundsMin;
    glm::vec4 boundsMax;
    glm::vec4 color;
    glm::vec4 uvScale;
    float     densityScale;
    uint32_t  sampleCount;
    uint32_t  enabled;
    uint32_t  frameIndex;
};

static_assert(sizeof(SmokeRenderPC) == 80);

} // namespace

SmokeRenderPass::SmokeRenderPass(Config config) : m_config(std::move(config))
{
    if (m_config.densityResourceName.empty())
    {
        throw std::invalid_argument("SmokeRenderPass: densityResourceName is required");
    }
    if (m_config.cameraBufferResourceName.empty())
    {
        throw std::invalid_argument("SmokeRenderPass: cameraBufferResourceName is required");
    }
    if (m_config.outputFormat == VK_FORMAT_UNDEFINED)
    {
        throw std::invalid_argument("SmokeRenderPass: outputFormat is required");
    }
}

void SmokeRenderPass::build(FrameGraph &frameGraph) const
{
    frameGraph.addPass("smoke_volume")
        .type(PassType::Fullscreen)
        .vertShader((paths::shaderDir / "fullscreen.vert.spv").string())
        .fragShader((paths::shaderDir / "smoke.frag.spv").string())
        .blend(BlendMode::Alpha)
        .pushConstantSize(sizeof(SmokeRenderPC), VK_SHADER_STAGE_FRAGMENT_BIT)
        .uniformBuffer(0, frameGraph.buffer(m_config.cameraBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(1, frameGraph.image(m_config.densityResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::clampToEdge())
        .sampledDepth(2, frameGraph.image(m_config.depthResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .colorAttachment(frameGraph.image(m_config.outputImage), m_config.outputFormat, VK_ATTACHMENT_LOAD_OP_LOAD)
        .execute([this](PassContext &context) {
            const SmokeRenderPC constants{
                .boundsMin    = glm::vec4(m_config.boundsMin, 0.0f),
                .boundsMax    = glm::vec4(m_config.boundsMax, 0.0f),
                .color        = glm::vec4(m_config.color, 1.0f),
                .uvScale      = glm::vec4(m_config.uvScale, 0.0f),
                .densityScale = std::max(m_config.densityScale, 0.0f),
                .sampleCount  = static_cast<uint32_t>(std::clamp(m_config.sampleCount, 8, 512)),
                .enabled      = m_config.enabled ? 1u : 0u,
                // Advances the output dither pattern so its noise averages out over time rather
                // than freezing into a fixed screen-space texture.
                .frameIndex   = m_frameIndex++,
            };
            context.cmd().pushConstants(context.pipelineLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, constants);
            context.cmd().draw(3);
        });
}

} // namespace lr::eulerian
