#include "core/passes/overlaylines/OverlayLinesPass.hpp"

#include "core/Paths.hpp"
#include "core/scene/Mesh.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

OverlayLinesPass::OverlayLinesPass(Config config, ResourceRegistry &registry)
    : m_config(std::move(config)), m_registry(&registry)
{
    if (m_config.maxLineCount == 0)
    {
        throw std::invalid_argument("OverlayLinesPass maxLineCount must be greater than zero");
    }

    const VkDeviceSize capacity = static_cast<VkDeviceSize>(m_config.maxLineCount) * 2 * sizeof(GpuVertex);
    m_registry->registerDynamicBuffer(m_config.vertexBufferResourceName, capacity, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
}

void OverlayLinesPass::setLines(const std::vector<OverlayLine> &lines)
{
    if (lines.size() > m_config.maxLineCount)
    {
        throw std::length_error("OverlayLinesPass line count exceeds configured capacity");
    }

    std::vector<GpuVertex> vertices;
    vertices.reserve(lines.size() * 2);
    for (const OverlayLine &line : lines)
    {
        const glm::vec4 colorAndOcclusion(line.style.color, line.style.occludedOpacity);
        vertices.push_back({glm::vec4(line.start, line.style.visibleOpacity), colorAndOcclusion});
        vertices.push_back({glm::vec4(line.end, line.style.visibleOpacity), colorAndOcclusion});
    }

    m_vertexCount = static_cast<uint32_t>(vertices.size());
    if (!vertices.empty())
    {
        m_registry->updateBuffer(m_config.vertexBufferResourceName, vertices.data(),
                                 static_cast<VkDeviceSize>(vertices.size() * sizeof(GpuVertex)));
    }
}

void OverlayLinesPass::build(FrameGraph &frameGraph)
{
    MeshLayout lineLayout;
    lineLayout.addPerVertexAttr<glm::vec4>("positionAndVisibleOpacity")
        .addPerVertexAttr<glm::vec4>("colorAndOccludedOpacity");

    GpuMeshLayout gpuLineLayout(lineLayout);
    gpuLineLayout.map("positionAndVisibleOpacity", 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT)
        .map("colorAndOccludedOpacity", 0, 1, VK_FORMAT_R32G32B32A32_SFLOAT);

    frameGraph.addPass("overlay_lines")
        .type(PassType::Geometry)
        .topology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST)
        .vertexLayout(gpuLineLayout)
        .vertShader((paths::shaderDir / "overlay_lines.vert.spv").string())
        .fragShader((paths::shaderDir / "overlay_lines.frag.spv").string())
        .uniformBuffer(0, frameGraph.buffer(m_config.cameraBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledDepth(1, frameGraph.image("gbufferDepth"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .vertexBuffer(0, frameGraph.buffer(m_config.vertexBufferResourceName))
        .colorAttachment(frameGraph.image("overlay"), VK_FORMAT_R16G16B16A16_SFLOAT, VK_ATTACHMENT_LOAD_OP_LOAD)
        .depthAttachment(frameGraph.image("overlayDepth"), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_LOAD)
        .execute([&](PassContext &context) {
            if (m_vertexCount > 0)
            {
                context.cmd().draw(m_vertexCount, 1);
            }
        });
}

} // namespace lr
