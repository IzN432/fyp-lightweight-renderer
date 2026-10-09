#include "core/passes/outline/OutlinePass.hpp"

#include "core/Paths.hpp"

#include <glm/vec4.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lr
{

namespace
{
struct OutlinePC
{
    glm::vec4 color;
    float     thickness;
    uint32_t  wordCount;
};

constexpr uint32_t bitsPerWord = 32;

uint32_t wordsFor(uint32_t ids) { return (ids + bitsPerWord - 1) / bitsPerWord; }
} // namespace

OutlinePass::OutlinePass(Config config, ResourceRegistry &registry) : m_config(std::move(config)), m_registry(&registry)
{
    if (m_config.maxPickingIds == 0)
    {
        throw std::invalid_argument("OutlinePass maxPickingIds must be greater than zero");
    }

    m_selectionWords.assign(wordsFor(m_config.maxPickingIds), 0u);
    const VkDeviceSize capacity = static_cast<VkDeviceSize>(m_selectionWords.size()) * sizeof(uint32_t);
    m_registry->registerDynamicBuffer(m_config.selectionBufferName, capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // The first frames may execute before any selection arrives, and the shader reads the buffer
    // unconditionally: start it empty rather than uninitialized.
    m_registry->updateBuffer(m_config.selectionBufferName, m_selectionWords.data(), capacity);
}

void OutlinePass::setSelection(const std::vector<uint32_t> &pickingIds)
{
    std::fill(m_selectionWords.begin(), m_selectionWords.end(), 0u);
    m_hasSelection = false;
    for (const uint32_t id : pickingIds)
    {
        if (id == 0 || id >= m_config.maxPickingIds)
        {
            continue;
        }
        m_selectionWords[id / bitsPerWord] |= 1u << (id % bitsPerWord);
        m_hasSelection = true;
    }

    // The whole bitset goes up every time. It is a few kilobytes, and sending it wholesale means a
    // shrinking selection cannot leave stale bits behind in the buffer's tail.
    m_registry->updateBuffer(m_config.selectionBufferName, m_selectionWords.data(),
                             static_cast<VkDeviceSize>(m_selectionWords.size() * sizeof(uint32_t)));
}

void OutlinePass::build(FrameGraph &fg)
{
    // Integer formats have no linear filtering, and the shader fetches exact texels anyway.
    const SamplerDesc pickingSampler{
        .magFilter    = VK_FILTER_NEAREST,
        .minFilter    = VK_FILTER_NEAREST,
        .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    };

    fg.addPass("outline")
        .type(PassType::Fullscreen)
        .vertShader((paths::shaderDir / "fullscreen.vert.spv").string())
        .fragShader((paths::shaderDir / "outline.frag.spv").string())
        .blend(BlendMode::Alpha)
        .pushConstantSize(sizeof(OutlinePC), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(0, fg.image(m_config.pickingImage), VK_SHADER_STAGE_FRAGMENT_BIT, pickingSampler)
        .storageBufferRead(1, fg.buffer(m_config.selectionBufferName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .colorAttachment(fg.image(m_config.outputImage), m_config.outputFormat, VK_ATTACHMENT_LOAD_OP_LOAD)
        .execute([this](PassContext &ctx) {
            if (!m_enabled || !m_hasSelection)
            {
                return;
            }
            const OutlinePC pc{
                .color     = glm::vec4(m_config.color, m_config.opacity),
                .thickness = m_config.thickness,
                .wordCount = static_cast<uint32_t>(m_selectionWords.size()),
            };
            ctx.cmd().pushConstants(ctx.pipelineLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, pc);
            ctx.cmd().draw(3);
        });
}

} // namespace lr
