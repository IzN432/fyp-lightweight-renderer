#include "core/passes/overlaypoints/OverlayPointsPass.hpp"

#include "core/Paths.hpp"

#include <glm/gtc/matrix_transform.hpp>

namespace lr
{

struct OverlayPointsPC
{
    glm::mat4 model           = glm::mat4(1.0f);
    float     pointSize       = 2.0f;
    float     occludedOpacity = 0.3f;
};

OverlayPointsPass::OverlayPointsPass(Config cfg) : m_cfg(std::move(cfg)) {}

void OverlayPointsPass::build(FrameGraph &fg, const GpuMeshLayout &layout) const
{
    auto pass = fg.addPass("overlay_points")
                    .type(PassType::Geometry)
                    .topology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
                    .vertexLayout(layout);

    pass.vertexBuffer(0, fg.buffer(m_cfg.pointsBufferResourceName));

    pass.vertShader((paths::shaderDir / "overlay_points.vert.spv").string())
        .fragShader((paths::shaderDir / "overlay_points.frag.spv").string())
        .pushConstantSize(sizeof(OverlayPointsPC), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .uniformBuffer(0, fg.buffer(m_cfg.cameraBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledDepth(1, fg.image("gbufferDepth"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .colorAttachment(fg.image("overlayPoints"), VK_FORMAT_R16G16B16A16_SFLOAT)
        .execute([&](CommandBuffer &cmd, VkPipelineLayout pipelineLayout) {
            if (!m_enabled)
            {
                return;
            }
            const OverlayPointsPC pc{.model =
                                         m_cfg.meshTransform ? m_cfg.meshTransform->localMatrix() : glm::mat4(1.0f)};
            for (size_t i = 0; i < m_cfg.pointsBufferUploadResult.singleMeshResults.size(); ++i)
            {
                const auto &vert = m_cfg.pointsBufferUploadResult.singleMeshResults[i];
                cmd.pushConstants(pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
                cmd.draw(m_cfg.vertexCounts[i], 1, vert.vertexOffset, 0);
            }
        });
}

} // namespace lr
