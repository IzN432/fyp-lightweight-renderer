#include "GeometryPass.hpp"

#include "core/Paths.hpp"

namespace lr
{

namespace
{
// Mirrors the push_constant block in geometry.frag — see the comment there for why this offset
// is needed once the pass draws more than one mesh.
struct GeometryPC
{
    uint32_t primitiveIdOffset;
};
}

GeometryPass::GeometryPass(Config cfg)
    : m_cfg(std::move(cfg))
{
}

void GeometryPass::build(FrameGraph &fg, const GpuMeshLayout &layout) const
{
    auto pass = fg.addPass("geometry")
        .type(PassType::Geometry)
        .vertexLayout(layout);

    for (const auto &[binding, bufferName] : m_cfg.vertexBufferResourceNames)
    {
        pass.vertexBuffer(binding, bufferName);
    }

    pass.indexBuffer(m_cfg.indexBufferResourceName)
        .vertShader((paths::shaderDir / "geometry.vert.spv").string())
        .fragShader((paths::shaderDir / "geometry.frag.spv").string())
        .pushConstantSize(sizeof(GeometryPC), VK_SHADER_STAGE_FRAGMENT_BIT)
        .bind({
            {
                .resourceName = m_cfg.cameraBufferResourceName,
                .binding      = 0,
                .type         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .stages       = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName    = m_cfg.diffuseTextureArrayResourceName,
                .binding         = 1,
                .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = m_cfg.materialCount,
                .stages          = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName    = m_cfg.normalTextureArrayResourceName,
                .binding         = 2,
                .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = m_cfg.materialCount,
                .stages          = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName    = m_cfg.metallicRoughnessTextureArrayResourceName,
                .binding         = 3,
                .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = m_cfg.materialCount,
                .stages          = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName    = m_cfg.emissiveTextureArrayResourceName,
                .binding         = 4,
                .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = m_cfg.materialCount,
                .stages          = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = m_cfg.faceGroupBufferResourceName,
                .binding      = 5,
                .type         = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName = m_cfg.materialBufferResourceName,
                .binding      = 6,
                .type         = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
        })
        .writes({
            {.name = "gbufferAlbedo",   .format = VK_FORMAT_R16G16B16A16_SFLOAT},
            {.name = "gbufferNormal",   .format = VK_FORMAT_R16G16_SFLOAT},
            {.name = "gbufferMaterial", .format = VK_FORMAT_R16G16B16A16_UNORM},
            {.name = "gbufferEmissive", .format = VK_FORMAT_R16G16B16A16_SFLOAT},
            {.name = "gbufferDepth",    .format = VK_FORMAT_D32_SFLOAT, .clearValue = {.depthStencil = {1.0f, 0}}},
        })
        .execute([&](CommandBuffer &cmd, VkPipelineLayout layout) {
            for (size_t i = 0; i < m_cfg.vertexBufferUploadResult.singleMeshResults.size(); ++i)
            {
                const auto &singleMesh = m_cfg.vertexBufferUploadResult.singleMeshResults[i];
                const auto &singleMeshIndex = m_cfg.indexBufferUploadResult.singleMeshResults[i];

                const GeometryPC pc{ .primitiveIdOffset = singleMeshIndex.firstIndex / 3 };
                cmd.pushConstants(layout, VK_SHADER_STAGE_FRAGMENT_BIT, pc);
                cmd.drawIndexed(singleMeshIndex.indexCount, 1, singleMeshIndex.firstIndex, singleMesh.vertexOffset, 0);
            }
        });
}

}  // namespace lr
