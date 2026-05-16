#include "core/passes/overlaygeometry/OverlayGeometryPass.hpp"

#include "core/overlay/OverlayMesh.hpp"
#include "core/Paths.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace lr
{

struct OverlayGeometryPC
{
    glm::mat4 model           = glm::mat4(1.0f);
    glm::vec3 color           = glm::vec3(1.0f, 0.0f, 1.0f);
    float     occludedOpacity = 0.0f;
    uint32_t  instanceIndex   = 0;  // 1-based; 0 written when not used (shouldn't happen)
};

OverlayGeometryPass::OverlayGeometryPass(Config cfg)
    : m_cfg(std::move(cfg))
{
}

void OverlayGeometryPass::uploadResources(ResourceRegistry &registry)
{
    std::vector<const Mesh*> meshes = {
        &OverlayMesh::cube().mesh(),
        &OverlayMesh::sphere().mesh(),
        &OverlayMesh::arrow().mesh(),
    };

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

    // Pre-register the picking image with TRANSFER_SRC so ImageReadback can copy it.
    // Must happen before FrameGraph::compile() to override the default usage flags.
    registry.registerImage(
        m_cfg.pickingImageName,
        VK_FORMAT_R32_UINT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}

void OverlayGeometryPass::setInstances(std::vector<OverlayInstance> instances)
{
    m_instances = std::move(instances);
}

void OverlayGeometryPass::build(FrameGraph &fg)
{
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
            {.name = "overlay",                  .format = VK_FORMAT_R16G16B16A16_SFLOAT},
            {.name = m_cfg.pickingImageName,     .format = VK_FORMAT_R32_UINT},
            {.name = "overlayDepth",             .format = VK_FORMAT_D32_SFLOAT, .clearValue = {.depthStencil = {1.0f, 0}}},
        })
        .execute([&](CommandBuffer &cmd, VkPipelineLayout pipelineLayout) {
            for (uint32_t i = 0; i < static_cast<uint32_t>(m_instances.size()); ++i)
            {
                const auto  &inst  = m_instances[i];
                const size_t idx   = static_cast<size_t>(inst.primitive);
                const auto  &vert  = m_vertexUpload.singleMeshResults[idx];
                const auto  &index = m_indexUpload.singleMeshResults[idx];

                const glm::mat4 T     = glm::translate(glm::mat4(1.0f), inst.position);
                const glm::quat q     = glm::quat(glm::radians(inst.eulerDegrees));
                const glm::mat4 model = T * glm::mat4_cast(q) * glm::scale(glm::mat4(1.0f), inst.scale);

                const glm::vec3 drawColor = (i == m_hoveredInstance)
                    ? glm::mix(inst.color, glm::vec3(1.0f), 0.4f)
                    : inst.color;

                OverlayGeometryPC pc{
                    .model           = model,
                    .color           = drawColor,
                    .occludedOpacity = inst.occludedOpacity,
                    .instanceIndex   = i + 1,  // 1-based; 0 = background in picking image
                };
                cmd.pushConstants(pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
                cmd.drawIndexed(index.indexCount, 1, index.firstIndex, vert.vertexOffset, 0);
            }
        });
}

}  // namespace lr
