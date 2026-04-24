#include "GeometryPass.hpp"

#include "core/upload/CameraUploader.hpp"

namespace lr
{

GeometryPass::GeometryPass(Config cfg)
    : m_cfg(std::move(cfg))
{
}

void GeometryPass::build(FrameGraph &fg, const GpuMeshLayout &layout) const
{
    const auto &shaderDir = m_cfg.shaderDir;

    fg.addPass("geometry")
        .type(PassType::Geometry)
        .vertexLayout(layout)
        .vertexBuffer(0, m_cfg.vertexAttributeBufferResourceName)
        .indexBuffer(m_cfg.indexBufferResourceName)
        .indexCount(m_cfg.indexCount)
        .vertShader((shaderDir / "geometry.vert.spv").string())
        .fragShader((shaderDir / "geometry.frag.spv").string())
        .bind({
            {
                .resourceName = CameraUploader::kBufferName,
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
            },
            {
                .resourceName    = m_cfg.normalTextureArrayResourceName,
                .binding         = 2,
                .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = m_cfg.materialCount,
                .stages          = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName    = m_cfg.metallicRoughnessTextureArrayResourceName,
                .binding         = 3,
                .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = m_cfg.materialCount,
                .stages          = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName    = m_cfg.emissiveTextureArrayResourceName,
                .binding         = 4,
                .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = m_cfg.materialCount,
                .stages          = VK_SHADER_STAGE_FRAGMENT_BIT,
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
            {.name = "gbufferMaterial", .format = VK_FORMAT_R8G8B8A8_UNORM},
            {.name = "swapchain",       .format = m_cfg.swapchainFormat},
            {.name = "gbufferDepth",    .format = VK_FORMAT_D32_SFLOAT, .clearValue = {.depthStencil = {1.0f, 0}}},
        })
        .execute([indexCount = m_cfg.indexCount](CommandBuffer &cmd, VkPipelineLayout) {
            cmd.drawIndexed(indexCount);
        });
}

}  // namespace lr
