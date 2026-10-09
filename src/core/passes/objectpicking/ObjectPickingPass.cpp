#include "ObjectPickingPass.hpp"

#include "core/Paths.hpp"

namespace lr
{
namespace
{
struct ObjectPickingPC
{
    glm::mat4 model;
    uint32_t  primitiveIdOffset;
    uint32_t  paletteOffset;
    uint32_t  skinEnabled;
    uint32_t  pickingId;
};
} // namespace

void ObjectPickingPass::uploadResources(ResourceRegistry &resources) const
{
    // SAMPLED_BIT is for OutlinePass, which derives the selection outline from these IDs rather
    // than redrawing the selected geometry.
    resources.registerImage(imageName, VK_FORMAT_R32_UINT,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                VK_IMAGE_USAGE_SAMPLED_BIT);
}

void ObjectPickingPass::build(FrameGraph &fg, const GpuMeshLayout &layout) const
{
    auto pass = fg.addPass("object picking").type(PassType::Geometry).vertexLayout(layout);
    for (const auto &[binding, bufferName] : m_config.vertexBufferResourceNames)
    {
        pass.vertexBuffer(binding, fg.buffer(bufferName));
    }

    pass.indexBuffer(fg.buffer(m_config.indexBufferResourceName))
        .vertShader((paths::shaderDir / "object_picking.vert.spv").string())
        .fragShader((paths::shaderDir / "object_picking.frag.spv").string())
        .cull(VK_CULL_MODE_NONE)
        .pushConstantSize(sizeof(ObjectPickingPC), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .uniformBuffer(0, fg.buffer(m_config.cameraBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledImageArray(1, fg.image(m_config.diffuseTextureArrayResourceName), m_config.materialCount,
                           VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(5, fg.buffer(m_config.faceGroupBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(6, fg.buffer(m_config.materialBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(7, fg.buffer(m_config.skinInfluenceEntriesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(8, fg.buffer(m_config.skinInfluenceOffsetsBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(9, fg.buffer(m_config.skinPositionIndicesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(10, fg.buffer(m_config.skinJointMatricesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .colorAttachment(fg.image(imageName), VK_FORMAT_R32_UINT, VK_ATTACHMENT_LOAD_OP_CLEAR,
                         {.color = {{0.0f, 0.0f, 0.0f, 0.0f}}})
        .depthAttachment(fg.image("objectPickingDepth"), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR,
                         {.depthStencil = {1.0f, 0}})
        .execute([this](PassContext &ctx) {
            for (size_t i = 0; i < m_config.draws.size(); ++i)
            {
                if (!m_config.draws.isLive(i))
                {
                    continue;
                }
                const SceneDraw       draw = m_config.draws.at(i, m_skinningEnabled);
                const ObjectPickingPC pc{
                    .model             = draw.model,
                    .primitiveIdOffset = draw.primitiveIdOffset,
                    .paletteOffset     = draw.paletteOffset,
                    .skinEnabled       = draw.skinEnabled,
                    .pickingId         = static_cast<uint32_t>(i + 1),
                };
                ctx.cmd().pushConstants(ctx.pipelineLayout(),
                                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
                ctx.cmd().drawIndexed(draw.indexCount, 1, draw.firstIndex, draw.vertexOffset, 0);
            }
        });
}

} // namespace lr
