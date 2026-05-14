#include "core/passes/overlaygeometry/OverlayGeometryPass.hpp"

#include "core/Paths.hpp"

#include <glm/gtc/matrix_transform.hpp>

namespace lr
{

struct OverlayGeometryPC
{
    glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(0.02f));
    glm::vec3 color = glm::vec3(1.0f, 0.0f, 1.0f);
    float occludedOpacity = 0.0f;
};

OverlayGeometryPass::OverlayGeometryPass(Config cfg)
    : m_cfg(std::move(cfg))
{
}

void OverlayGeometryPass::uploadResources(ResourceRegistry &registry, const std::vector<const Mesh*> &meshes)
{
    MeshUploader meshUploader(registry);
    m_vertexUpload = meshUploader.uploadVertexBuffer(meshes, {
        .vertexBufferName     = m_cfg.vertexBufferName,
        .vertexAttributeNames = { "normal", "color" },
        .includePosition      = true,
    });
    m_indexUpload = meshUploader.uploadIndexBuffer(meshes, {
        .indexBufferName = m_cfg.indexBufferName,
    });

    m_gpuMeshLayout = GpuMeshLayout(meshes[0]->layout());
    m_gpuMeshLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT);
    m_gpuMeshLayout.map("normal", 0, 1, VK_FORMAT_R32G32B32_SFLOAT);
    m_gpuMeshLayout.map("color",  0, 2, VK_FORMAT_R32G32B32_SFLOAT);
}

void OverlayGeometryPass::build(FrameGraph &fg) const
{
    const OverlayGeometryPC overlayGeometryPC{
        .color = glm::vec3(1.0f, 0.0f, 0.0f),
        .occludedOpacity = 0.1f
    };

    auto pass = fg.addPass("overlay_geometry")
                    .type(PassType::Geometry)
                    .vertexLayout(m_gpuMeshLayout);

    pass.vertexBuffer(0, m_cfg.vertexBufferName);

    pass.indexBuffer(m_cfg.indexBufferName)
        .vertShader((paths::shaderDir / "overlay_geometry.vert.spv").string())
        .fragShader((paths::shaderDir / "overlay_geometry.frag.spv").string())
        .pushConstantSize(sizeof(OverlayGeometryPC), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .bind({
            {
                .resourceName = m_cfg.cameraBufferResourceName,
                .binding      = 0,
                .type         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .stages       = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName = "gbufferDepth",
                .binding      = 1,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
            }
        })
        .writes({
            {.name = "overlay",      .format = VK_FORMAT_R16G16B16A16_SFLOAT},
            {.name = "overlayDepth", .format = VK_FORMAT_D32_SFLOAT, .clearValue = {.depthStencil = {1.0f, 0}}},
        })
        .execute([&, overlayGeometryPC](CommandBuffer &cmd, VkPipelineLayout pipelineLayout) {
            for (size_t i = 0; i < m_vertexUpload.singleMeshResults.size(); ++i)
            {
                const auto &vert  = m_vertexUpload.singleMeshResults[i];
                const auto &index = m_indexUpload.singleMeshResults[i];
                cmd.pushConstants(pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, overlayGeometryPC);
                cmd.drawIndexed(index.indexCount, 1, index.firstIndex, vert.vertexOffset, 0);
            }
        });
}

}  // namespace lr
