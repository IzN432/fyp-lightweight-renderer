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

OverlayPointsPass::OverlayPointsPass(Config cfg)
    : m_cfg(std::move(cfg))
{
}

void OverlayPointsPass::build(FrameGraph &fg, const GpuMeshLayout &layout) const
{
    auto pass = fg.addPass("overlay_points")
                    .type(PassType::Geometry)
                    .topology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
                    .vertexLayout(layout);

    pass.vertexBuffer(0, m_cfg.positionBufferResourceName);
    pass.vertexBuffer(1, m_cfg.colorBufferResourceName);

    pass.vertShader((paths::shaderDir / "overlay_points.vert.spv").string())
        .fragShader((paths::shaderDir / "overlay_points.frag.spv").string())
        .pushConstantSize(sizeof(OverlayPointsPC), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .bind({
            {
                .resourceName = m_cfg.cameraBufferResourceName,
                .binding      = 0,
                .type         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .stages       = VK_SHADER_STAGE_VERTEX_BIT,
            },
            {
                .resourceName = "gbufferDepth",
                .binding      = 1,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
        })
        .writes({
            {.name = "overlayPoints", .format = VK_FORMAT_R16G16B16A16_SFLOAT},
        })
        .execute([&](CommandBuffer &cmd, VkPipelineLayout pipelineLayout) {
            if (!m_enabled) return;
            const OverlayPointsPC pc{};
            for (size_t i = 0; i < m_cfg.positionBufferUploadResult.singleMeshResults.size(); ++i)
            {
                const auto &vert = m_cfg.positionBufferUploadResult.singleMeshResults[i];
                cmd.pushConstants(pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
                cmd.draw(m_cfg.vertexCounts[i], 1, vert.vertexOffset, 0);
            }
        });
}

}  // namespace lr
