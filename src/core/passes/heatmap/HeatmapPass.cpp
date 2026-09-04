#include "HeatmapPass.hpp"

#include "core/Paths.hpp"

namespace lr
{

namespace
{
struct HeatmapPC
{
    glm::mat4 model;
};
} // namespace

HeatmapPass::HeatmapPass(Config cfg) : m_cfg(std::move(cfg)) {}

void HeatmapPass::build(FrameGraph &fg, const GpuMeshLayout &layout) const
{
    fg.addPass("heatmap")
        .type(PassType::Geometry)
        .vertexLayout(layout)
        .vertexBuffer(0, fg.buffer(m_cfg.vertexBufferResourceName))
        .indexBuffer(fg.buffer(m_cfg.indexBufferResourceName))
        .vertShader((paths::shaderDir / "heatmap.vert.spv").string())
        .fragShader((paths::shaderDir / "heatmap.frag.spv").string())
        .pushConstantSize(sizeof(HeatmapPC), VK_SHADER_STAGE_VERTEX_BIT)
        .uniformBuffer(0, fg.buffer(m_cfg.cameraBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .colorAttachment(fg.image("heatmap"), VK_FORMAT_R16G16B16A16_SFLOAT)
        .depthAttachment(fg.image("heatmapDepth"), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR,
                         {.depthStencil = {1.0f, 0}})
        .execute([&](PassContext &ctx) {
            if (!m_enabled)
            {
                return;
            }

            // Only the main mesh (singleMeshResults[0]) is drawn — light visuals never get a
            // "color" attribute (see SceneManager::updateMainMeshHeatmapBuffer), so they aren't
            // part of m_cfg.vertexBufferUploadResult/indexBufferUploadResult to begin with.
            const auto &vert  = m_cfg.vertexBufferUploadResult.singleMeshResults[0];
            const auto &index = m_cfg.indexBufferUploadResult.singleMeshResults[0];

            const HeatmapPC pc{
                .model = m_cfg.meshTransform ? m_cfg.meshTransform->localMatrix() : glm::mat4(1.0f),
            };
            ctx.cmd().pushConstants(ctx.pipelineLayout(), VK_SHADER_STAGE_VERTEX_BIT, pc);
            ctx.cmd().drawIndexed(index.indexCount, 1, index.firstIndex, vert.vertexOffset, 0);
        });
}

} // namespace lr
