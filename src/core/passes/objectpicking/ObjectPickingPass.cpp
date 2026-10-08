#include "ObjectPickingPass.hpp"

#include "core/Paths.hpp"
#include "core/scene/Scene.hpp"

#include <stdexcept>

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
    resources.registerImage(imageName, VK_FORMAT_R32_UINT,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}

void ObjectPickingPass::setSceneGeometry(VertexBufferUploadResult vertices, IndexBufferUploadResult indices,
                                         std::vector<const TransformComponent *> transforms,
                                         std::vector<SceneObject *> objects, std::vector<SkinDrawInfo> skins)
{
    if (vertices.singleMeshResults.size() != indices.singleMeshResults.size() ||
        vertices.singleMeshResults.size() != transforms.size() ||
        vertices.singleMeshResults.size() != objects.size() ||
        vertices.singleMeshResults.size() != skins.size())
    {
        throw std::invalid_argument("ObjectPickingPass::setSceneGeometry: draw arrays must be parallel");
    }
    m_config.vertexBufferUploadResult = std::move(vertices);
    m_config.indexBufferUploadResult  = std::move(indices);
    m_config.meshTransforms           = std::move(transforms);
    m_config.meshObjects              = std::move(objects);
    m_config.skinDrawInfos            = std::move(skins);
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
            for (size_t i = 0; i < m_config.vertexBufferUploadResult.singleMeshResults.size(); ++i)
            {
                SceneObject *object = m_config.meshObjects[i];
                if (!object->scene().contains(object->id()))
                {
                    continue;
                }
                const auto &mesh      = m_config.vertexBufferUploadResult.singleMeshResults[i];
                const auto &range     = m_config.indexBufferUploadResult.singleMeshResults[i];
                const auto *transform = m_config.meshTransforms[i];
                const auto &skin      = m_config.skinDrawInfos[i];
                const ObjectPickingPC pc{
                    .model             = transform ? transform->worldMatrix() : glm::mat4(1.0f),
                    .primitiveIdOffset = range.firstIndex / 3,
                    .paletteOffset     = skin.paletteOffset,
                    .skinEnabled       = skin.skinEnabled ? 1u : 0u,
                    .pickingId         = static_cast<uint32_t>(i + 1),
                };
                ctx.cmd().pushConstants(ctx.pipelineLayout(),
                                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
                ctx.cmd().drawIndexed(range.indexCount, 1, range.firstIndex, mesh.vertexOffset, 0);
            }
        });
}

} // namespace lr
