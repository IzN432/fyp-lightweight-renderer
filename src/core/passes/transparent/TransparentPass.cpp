#include "TransparentPass.hpp"

#include "core/Paths.hpp"
#include "core/scene/Scene.hpp"

#include <glm/geometric.hpp>

#include <algorithm>
#include <stdexcept>

namespace lr
{

namespace
{
// Mirrors the push_constant block in transparent.frag; the first four members are geometry.vert's.
struct TransparentPC
{
    glm::mat4 model;
    uint32_t  primitiveIdOffset;
    uint32_t  paletteOffset;
    uint32_t  skinEnabled;
    uint32_t  numLights;
    uint32_t  pfMips;
};

constexpr VkShaderStageFlags kVertFrag = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
} // namespace

TransparentPass::TransparentPass(Config cfg) : m_cfg(std::move(cfg))
{
    if (!m_cfg.isBlendMaterial)
    {
        throw std::invalid_argument("TransparentPass: Config::isBlendMaterial is required");
    }
}

void TransparentPass::setSceneGeometry(VertexBufferUploadResult vertices, IndexBufferUploadResult indices,
                                       std::vector<const TransformComponent *> transforms,
                                       std::vector<SceneObject *> objects, std::vector<SkinDrawInfo> skins,
                                       const std::vector<const Mesh *> &meshes)
{
    const size_t count = vertices.singleMeshResults.size();
    if (indices.singleMeshResults.size() != count || transforms.size() != count || objects.size() != count ||
        skins.size() != count || meshes.size() != count)
    {
        throw std::invalid_argument("TransparentPass::setSceneGeometry: draw arrays must be parallel");
    }

    m_draws.clear();
    m_draws.reserve(count);
    for (const Mesh *mesh : meshes)
    {
        DrawBounds  draw;
        const auto &positions = mesh->positions();
        if (!positions.empty())
        {
            glm::vec3 low = positions.front(), high = positions.front();
            for (const glm::vec3 &position : positions)
            {
                low  = glm::min(low, position);
                high = glm::max(high, position);
            }
            draw.localCenter = (low + high) * 0.5f;
        }
        draw.materials = mesh->faceGroups();
        std::ranges::sort(draw.materials);
        const auto duplicates = std::ranges::unique(draw.materials);
        draw.materials.erase(duplicates.begin(), duplicates.end());
        m_draws.push_back(std::move(draw));
    }

    m_cfg.geometry.vertexBufferUploadResult = std::move(vertices);
    m_cfg.geometry.indexBufferUploadResult  = std::move(indices);
    m_cfg.geometry.meshTransforms           = std::move(transforms);
    m_cfg.geometry.meshObjects              = std::move(objects);
    m_cfg.geometry.skinDrawInfos            = std::move(skins);
}

void TransparentPass::build(FrameGraph &fg, const GpuMeshLayout &layout) const
{
    const GeometryPass::Config &geometry = m_cfg.geometry;
    auto                        pass     = fg.addPass("transparent").type(PassType::Geometry).vertexLayout(layout);

    for (const auto &[binding, bufferName] : geometry.vertexBufferResourceNames)
    {
        pass.vertexBuffer(binding, fg.buffer(bufferName));
    }

    pass.indexBuffer(fg.buffer(geometry.indexBufferResourceName))
        .vertShader((paths::shaderDir / "geometry.vert.spv").string())
        .fragShader((paths::shaderDir / "transparent.frag.spv").string())
        .cull(VK_CULL_MODE_NONE) // back faces of single-sided materials are discarded in the shader
        // Blending each surface over what is behind it needs the draws in back-to-front order (see execute).
        .blend(BlendMode::PremultipliedAlpha)
        // Test against the opaque depth, but don't write it: transparent surfaces don't hide each other.
        // LESS_OR_EQUAL keeps decals coplanar with the surface they sit on.
        .depth(true, false, VK_COMPARE_OP_LESS_OR_EQUAL)
        .pushConstantSize(sizeof(TransparentPC), kVertFrag)
        .uniformBuffer(0, fg.buffer(geometry.cameraBufferResourceName), kVertFrag)
        .sampledImageArray(1, fg.image(geometry.diffuseTextureArrayResourceName), geometry.materialCount,
                           VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImageArray(2, fg.image(geometry.normalTextureArrayResourceName), geometry.materialCount,
                           VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImageArray(3, fg.image(geometry.metallicRoughnessTextureArrayResourceName), geometry.materialCount,
                           VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImageArray(4, fg.image(geometry.emissiveTextureArrayResourceName), geometry.materialCount,
                           VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(5, fg.buffer(geometry.faceGroupBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(6, fg.buffer(geometry.materialBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(7, fg.buffer(geometry.skinInfluenceEntriesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(8, fg.buffer(geometry.skinInfluenceOffsetsBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(9, fg.buffer(geometry.skinPositionIndicesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(10, fg.buffer(geometry.skinJointMatricesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(11, fg.buffer(m_cfg.lightBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(12, fg.image("ibl_irradiance"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(13, fg.image("ibl_prefiltered"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(14, fg.image("ibl_brdf_lut"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(15, fg.image("ltc1"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(16, fg.image("ltc2"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledDepth(17, fg.image(m_cfg.spotShadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::shadowComparison())
        .uniformBuffer(18, fg.buffer(m_cfg.spotShadowParamsBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledDepth(19, fg.image(m_cfg.cascadedShadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::shadowComparison())
        .uniformBuffer(20, fg.buffer(m_cfg.cascadedShadowParamsBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        // Both shadow maps a second time, uncompared, for the PCSS blocker search.
        .sampledDepth(21, fg.image(m_cfg.spotShadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::depthFetch())
        .sampledDepth(22, fg.image(m_cfg.cascadedShadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::depthFetch())
        .sampledDepth(23, fg.image(m_cfg.areaShadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::shadowComparison())
        .uniformBuffer(24, fg.buffer(m_cfg.areaShadowParamsBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledDepth(25, fg.image(m_cfg.areaShadowImageResourceName), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::depthFetch())
        // GeometryPass's sample count, so the depth below is its unresolved, per-sample depth.
        .samples(VK_SAMPLE_COUNT_4_BIT)
        .colorAttachment(fg.image(m_cfg.outputImage), VK_FORMAT_R16G16B16A16_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR,
                         {.color = {{0.0f, 0.0f, 0.0f, 0.0f}}})
        .depthAttachment(fg.image("gbufferDepth"), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_LOAD)
        // Reads m_cfg/m_draws when the pass runs, so geometry, light count and material edits take effect
        // without rebuilding the graph.
        .execute([this](PassContext &ctx) {
            const GeometryPass::Config &geometry = m_cfg.geometry;
            const glm::vec3             eye      = m_cfg.eyePosition ? m_cfg.eyePosition() : glm::vec3(0.0f);

            m_order.clear();
            for (size_t i = 0; i < m_draws.size(); ++i)
            {
                SceneObject *object = geometry.meshObjects[i];
                if (!object->scene().contains(object->id()) ||
                    !std::ranges::any_of(m_draws[i].materials, m_cfg.isBlendMaterial))
                {
                    continue;
                }
                const TransformComponent *transform = geometry.meshTransforms[i];
                const glm::vec3           center =
                    transform ? glm::vec3(transform->worldMatrix() * glm::vec4(m_draws[i].localCenter, 1.0f))
                              : m_draws[i].localCenter;
                const glm::vec3 offset = center - eye;
                m_order.emplace_back(glm::dot(offset, offset), i);
            }
            // Farthest first, so each surface blends over the ones behind it.
            std::ranges::sort(m_order, std::greater<>{}, &std::pair<float, size_t>::first);

            for (const auto &[distance, i] : m_order)
            {
                const auto               &mesh      = geometry.vertexBufferUploadResult.singleMeshResults[i];
                const auto               &range     = geometry.indexBufferUploadResult.singleMeshResults[i];
                const TransformComponent *transform = geometry.meshTransforms[i];
                const SkinDrawInfo       &skin      = geometry.skinDrawInfos[i];
                const TransparentPC       pc{
                    .model             = transform ? transform->worldMatrix() : glm::mat4(1.0f),
                    .primitiveIdOffset = range.firstIndex / 3,
                    .paletteOffset     = skin.paletteOffset,
                    .skinEnabled       = skin.skinEnabled && m_skinningEnabled ? 1u : 0u,
                    .numLights         = m_cfg.numLights,
                    .pfMips            = m_cfg.pfMips,
                };
                ctx.cmd().pushConstants(ctx.pipelineLayout(), kVertFrag, pc);
                ctx.cmd().drawIndexed(range.indexCount, 1, range.firstIndex, mesh.vertexOffset, 0);
            }
        });
}

} // namespace lr
