#include "GeometryPass.hpp"

#include "core/Paths.hpp"

namespace lr
{

namespace
{
// Mirrors the push_constant block in geometry.vert/geometry.frag. model is consumed by the vertex
// shader; primitiveIdOffset is consumed by the fragment shader (see its comment for why it's
// needed once the pass draws more than one mesh) — both shaders declare the full struct so their
// offsets agree, even though each only reads its own field(s).
struct GeometryPC
{
    glm::mat4 model;
    uint32_t  primitiveIdOffset;
    uint32_t  paletteOffset;
    uint32_t  skinEnabled;
};
} // namespace

GeometryPass::GeometryPass(Config cfg) : m_cfg(std::move(cfg)) {}

void GeometryPass::build(FrameGraph &fg, const GpuMeshLayout &layout) const
{
    auto pass = fg.addPass("geometry").type(PassType::Geometry).vertexLayout(layout);

    for (const auto &[binding, bufferName] : m_cfg.vertexBufferResourceNames)
    {
        pass.vertexBuffer(binding, fg.buffer(bufferName));
    }

    pass.indexBuffer(fg.buffer(m_cfg.indexBufferResourceName))
        .vertShader((paths::shaderDir / "geometry.vert.spv").string())
        .fragShader((paths::shaderDir / "geometry.frag.spv").string())
        .pushConstantSize(sizeof(GeometryPC), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .uniformBuffer(0, fg.buffer(m_cfg.cameraBufferResourceName),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImageArray(1, fg.image(m_cfg.diffuseTextureArrayResourceName), m_cfg.materialCount,
                           VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImageArray(2, fg.image(m_cfg.normalTextureArrayResourceName), m_cfg.materialCount,
                           VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImageArray(3, fg.image(m_cfg.metallicRoughnessTextureArrayResourceName), m_cfg.materialCount,
                           VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImageArray(4, fg.image(m_cfg.emissiveTextureArrayResourceName), m_cfg.materialCount,
                           VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(5, fg.buffer(m_cfg.faceGroupBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(6, fg.buffer(m_cfg.materialBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(7, fg.buffer(m_cfg.skinInfluenceEntriesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(8, fg.buffer(m_cfg.skinInfluenceOffsetsBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(9, fg.buffer(m_cfg.skinPositionIndicesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(10, fg.buffer(m_cfg.skinJointMatricesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .colorAttachment(fg.image("gbufferAlbedo"), VK_FORMAT_R16G16B16A16_SFLOAT)
        .colorAttachment(fg.image("gbufferNormal"), VK_FORMAT_R16G16_SFLOAT)
        .colorAttachment(fg.image("gbufferMaterial"), VK_FORMAT_R16G16B16A16_UNORM)
        .colorAttachment(fg.image("gbufferEmissive"), VK_FORMAT_R16G16B16A16_SFLOAT)
        .depthAttachment(fg.image("gbufferDepth"), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR,
                         {.depthStencil = {1.0f, 0}})
        .execute([&](PassContext &ctx) {
            for (size_t i = 0; i < m_cfg.vertexBufferUploadResult.singleMeshResults.size(); ++i)
            {
                const auto &singleMesh      = m_cfg.vertexBufferUploadResult.singleMeshResults[i];
                const auto &singleMeshIndex = m_cfg.indexBufferUploadResult.singleMeshResults[i];

                const TransformComponent *transform = m_cfg.meshTransforms[i];
                const glm::mat4 model = transform ? transform->worldMatrix() : glm::mat4(1.0f);

                const SkinDrawInfo &skin = m_cfg.skinDrawInfos[i];
                const GeometryPC pc{.model             = model,
                                    .primitiveIdOffset = singleMeshIndex.firstIndex / 3,
                                    .paletteOffset     = skin.paletteOffset,
                                    .skinEnabled       = skin.skinEnabled && m_skinningEnabled ? 1u : 0u};
                ctx.cmd().pushConstants(ctx.pipelineLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                        pc);
                ctx.cmd().drawIndexed(singleMeshIndex.indexCount, 1, singleMeshIndex.firstIndex,
                                      singleMesh.vertexOffset, 0);
            }
        });
}

} // namespace lr
