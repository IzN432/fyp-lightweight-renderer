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
    glm::vec4 coolColor;
    glm::vec4 warmColor;
    glm::vec4 uvScale;
    float     densityScale;
    float     coolTemperature;
    float     warmTemperature;
    uint32_t  sampleCount;
    uint32_t  enabled;
    uint32_t  frameIndex;
};

static_assert(sizeof(SmokeRenderPC) == 112);

} // namespace

SmokeRenderPass::SmokeRenderPass(Config config) : m_config(std::move(config))
{
    if (m_config.densityResourceName.empty())
    {
        throw std::invalid_argument("SmokeRenderPass: densityResourceName is required");
    }
    if (m_config.temperatureResourceName.empty())
    {
        throw std::invalid_argument("SmokeRenderPass: temperatureResourceName is required");
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
        .vertShader((paths::shaderDirectory() / "fullscreen.vert.spv").string())
        .fragShader((paths::shaderDirectory() / "smoke.frag.spv").string())
        .blend(BlendMode::Alpha)
        .pushConstantSize(sizeof(SmokeRenderPC), VK_SHADER_STAGE_FRAGMENT_BIT)
        .uniformBuffer(0, frameGraph.buffer(m_config.cameraBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(1, frameGraph.image(m_config.densityResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::clampToEdge())
        .sampledDepth(2, frameGraph.image(m_config.depthResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(3, frameGraph.image(m_config.temperatureResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::clampToEdge())
        .colorAttachment(frameGraph.image(m_config.outputImage), m_config.outputFormat, VK_ATTACHMENT_LOAD_OP_LOAD)
        .execute([this](PassContext &context) {
            const SmokeRenderPC constants{
                .boundsMin    = glm::vec4(m_config.boundsMin, 0.0f),
                .boundsMax    = glm::vec4(m_config.boundsMax, 0.0f),
                .coolColor    = glm::vec4(m_config.coolColor, 1.0f),
                .warmColor    = glm::vec4(m_config.warmColor, 1.0f),
                .uvScale      = glm::vec4(m_config.uvScale, 0.0f),
                .densityScale = std::max(m_config.densityScale, 0.0f),
                .coolTemperature = m_config.coolTemperature,
                // Kept strictly above the cool end so the ramp always has a non-degenerate span;
                // the shader guards too, but clamping here keeps the two ends from crossing.
                .warmTemperature = std::max(m_config.warmTemperature, m_config.coolTemperature + 1e-4f),
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
