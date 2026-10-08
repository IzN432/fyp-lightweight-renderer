#include "AreaShadowPass.hpp"

#include "core/Paths.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lr
{
namespace
{
// A quad with no extent would put the eye on the quad's own plane, so both the eye offset and the
// emitter radius are floored.
constexpr float kMinHalfExtent = 1e-3f;

struct ShadowPC
{
    glm::mat4 model;
    uint32_t  primitiveIdOffset;
    uint32_t  paletteOffset;
    uint32_t  skinEnabled;
};
} // namespace

AreaShadowPass::AreaShadowPass(ResourceRegistry &resources, Config cfg)
    : m_resources(resources), m_cfg(std::move(cfg))
{
    if (m_cfg.resolution == 0 || m_cfg.range <= 0.0f)
    {
        throw std::invalid_argument("AreaShadowPass: invalid configuration");
    }
    resources.registerPersistentImageArray(m_cfg.shadowImage, VK_FORMAT_D32_SFLOAT,
                                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                           {m_cfg.resolution, m_cfg.resolution}, AreaShadowGpuData::maxLayers,
                                           VK_IMAGE_ASPECT_DEPTH_BIT);
    resources.registerDynamicBuffer(m_cfg.paramsBuffer, sizeof(AreaShadowGpuData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    const AreaShadowGpuData disabled{};
    resources.updateBuffer(m_cfg.paramsBuffer, &disabled, sizeof(disabled));
}

AreaShadowGpuData AreaShadowPass::shadowData() const
{
    AreaShadowGpuData result{};
    for (SceneObject *object : m_cfg.lightObjects)
    {
        if (!object || !object->scene().contains(object->id()) || !object->hasComponent<Light>())
            continue;
        const auto *area = std::get_if<AreaLight>(&object->getComponent<Light>().light);
        if (!area)
            continue;

        const float halfFovDegrees = std::clamp(area->spreadAngleDegrees, 1.0f, 89.0f);
        const float tanHalfFov = std::tan(glm::radians(halfFovDegrees));

        const Transform &transform = object->getComponent<TransformComponent>().transform();
        const glm::vec3  position  = transform.position();
        const glm::vec3  forward   = transform.forward();
        const glm::vec3  up        = transform.up();

        const float halfWidth  = std::max(area->size.x * 0.5f, kMinHalfExtent);
        const float halfHeight = std::max(area->size.y * 0.5f, kMinHalfExtent);

        // Pull the eye back until the frustum exactly inscribes the quad at the quad's own plane, so
        // coverage then widens with distance in front of it the way the lit region does. Deriving the
        // offset from the quad rather than exposing it as a knob is also what keeps PCSS
        // well-conditioned: the near plane below scales with the emitter, so the UV light size stays
        // around 0.5 instead of blowing up the way a small hand-set near plane does for spot lights.
        const float eyeOffset = std::max(halfWidth, halfHeight) / tanHalfFov;
        // Just past the quad's plane, which clips the light's own emissive visual quad (SceneGpu puts
        // it in the same draw list) and anything else coplanar with the emitter. For a one-sided light
        // that is also exactly the right cull: nothing behind the quad may shadow what is in front.
        const float nearPlane = eyeOffset * 1.01f;
        const float farPlane  = eyeOffset + m_cfg.range;

        glm::mat4 projection = glm::perspective(glm::radians(halfFovDegrees * 2.0f), 1.0f, nearPlane, farPlane);
        projection[1][1] *= -1.0f;
        glm::mat4 clip(1.0f);
        clip[2][2] = 0.5f;
        clip[3][2] = 0.5f;

        // A disc of the same area as the quad: PCSS filters with a disc, so a rectangle's penumbra is
        // anisotropic in a way the kernel cannot express, and equal area is the least-wrong radius.
        const float emitterRadius = std::sqrt(halfWidth * halfHeight);

        const uint32_t lightIndex = result.header.x++;
        // Both faces are always built and drawn, even for a one-sided light. That keeps the layers
        // contiguous (so instanced rendering can map gl_InstanceIndex straight to gl_Layer) and every
        // matrix valid; the unused face costs one extra draw and is never sampled for anything that
        // receives light.
        for (uint32_t side = 0; side < AreaShadowGpuData::sidesPerLight; ++side)
        {
            const glm::vec3 direction = side == 0 ? forward : -forward;
            const glm::vec3 eye       = position - direction * eyeOffset;
            const glm::mat4 view      = glm::lookAt(eye, eye + direction, up);
            const uint32_t  layer     = lightIndex * AreaShadowGpuData::sidesPerLight + side;
            result.lightViewProj[layer] = clip * projection * view;
            result.pcss[layer] = glm::vec4(emitterRadius / (2.0f * tanHalfFov * nearPlane), nearPlane,
                                           farPlane, 0.0f);
        }

        if (result.header.x == AreaShadowGpuData::maxLights)
            break;
    }
    result.header.y = AreaShadowGpuData::sidesPerLight;
    return result;
}

void AreaShadowPass::build(FrameGraph &fg, const GpuMeshLayout &layout) const
{
    auto pass = fg.addPass("areaShadow").type(PassType::Geometry).vertexLayout(layout);
    for (const auto &[binding, name] : m_cfg.geometry.vertexBufferResourceNames)
        pass.vertexBuffer(binding, fg.buffer(name));

    pass.indexBuffer(fg.buffer(m_cfg.geometry.indexBufferResourceName))
        .vertShader((paths::shaderDir / "area_shadow.vert.spv").string())
        // The fragment stage is the same alpha-and-facing test as the spot pass; only the projection
        // differs, and that lives in the vertex stage's params buffer.
        .fragShader((paths::shaderDir / "spot_shadow.frag.spv").string())
        .cull(VK_CULL_MODE_BACK_BIT)
        .depthBias(1.25f, 1.75f)
        .renderingLayers(AreaShadowGpuData::maxLayers)
        .pushConstantSize(sizeof(ShadowPC), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .uniformBuffer(0, fg.buffer(m_cfg.paramsBuffer), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledImageArray(1, fg.image(m_cfg.geometry.diffuseTextureArrayResourceName), m_cfg.geometry.materialCount,
                           VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(5, fg.buffer(m_cfg.geometry.faceGroupBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(6, fg.buffer(m_cfg.geometry.materialBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(7, fg.buffer(m_cfg.geometry.skinInfluenceEntriesBufferResourceName),
                           VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(8, fg.buffer(m_cfg.geometry.skinInfluenceOffsetsBufferResourceName),
                           VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(9, fg.buffer(m_cfg.geometry.skinPositionIndicesBufferResourceName),
                           VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(10, fg.buffer(m_cfg.geometry.skinJointMatricesBufferResourceName),
                           VK_SHADER_STAGE_VERTEX_BIT)
        .depthAttachment(fg.image(m_cfg.shadowImage), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR,
                         {.depthStencil = {1.0f, 0}}, ExtentSpec::absolute(m_cfg.resolution, m_cfg.resolution))
        .execute([this](PassContext &ctx) {
            const AreaShadowGpuData data = shadowData();
            m_resources.updateBuffer(m_cfg.paramsBuffer, &data, sizeof(data));
            const uint32_t layerCount = data.header.x * data.header.y;
            if (layerCount == 0)
                return;
            const SceneDrawList &draws = m_cfg.geometry.draws;
            for (size_t i = 0; i < draws.size(); ++i)
            {
                if (!draws.isLive(i))
                    continue;
                const SceneDraw draw = draws.at(i);
                const ShadowPC  pc{.model             = draw.model,
                                   .primitiveIdOffset = draw.primitiveIdOffset,
                                   .paletteOffset     = draw.paletteOffset,
                                   .skinEnabled       = draw.skinEnabled};
                ctx.cmd().pushConstants(ctx.pipelineLayout(),
                                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
                ctx.cmd().drawIndexed(draw.indexCount, layerCount, draw.firstIndex, draw.vertexOffset, 0);
            }
        });
}

} // namespace lr
