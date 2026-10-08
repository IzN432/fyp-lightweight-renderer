#include "CascadedShadowPass.hpp"

#include "core/Paths.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace lr
{
namespace
{
struct ShadowPC
{
    glm::mat4 model;
    uint32_t primitiveIdOffset;
    uint32_t paletteOffset;
    uint32_t skinEnabled;
};

std::array<glm::vec3, 8> frustumSliceCorners(const Camera &camera, const Transform &transform, float aspect,
                                             float nearDistance, float farDistance)
{
    std::array<glm::vec3, 8> corners{};
    const glm::vec3 forward = transform.forward();
    const glm::vec3 right = transform.right();
    const glm::vec3 up = transform.up();
    const glm::vec3 origin = transform.position();
    float nearHalfHeight;
    float farHalfHeight;
    if (camera.projectionType == ProjectionType::Perspective)
    {
        const float tanHalfFov = std::tan(glm::radians(camera.fovYDegrees) * 0.5f);
        nearHalfHeight = nearDistance * tanHalfFov;
        farHalfHeight = farDistance * tanHalfFov;
    } else
    {
        nearHalfHeight = camera.orthoHeight * 0.5f;
        farHalfHeight = nearHalfHeight;
    }
    const float nearHalfWidth = nearHalfHeight * aspect;
    const float farHalfWidth = farHalfHeight * aspect;
    const glm::vec3 nearCenter = origin + forward * nearDistance;
    const glm::vec3 farCenter = origin + forward * farDistance;
    corners[0] = nearCenter - right * nearHalfWidth - up * nearHalfHeight;
    corners[1] = nearCenter + right * nearHalfWidth - up * nearHalfHeight;
    corners[2] = nearCenter + right * nearHalfWidth + up * nearHalfHeight;
    corners[3] = nearCenter - right * nearHalfWidth + up * nearHalfHeight;
    corners[4] = farCenter - right * farHalfWidth - up * farHalfHeight;
    corners[5] = farCenter + right * farHalfWidth - up * farHalfHeight;
    corners[6] = farCenter + right * farHalfWidth + up * farHalfHeight;
    corners[7] = farCenter - right * farHalfWidth + up * farHalfHeight;
    return corners;
}
} // namespace

CascadedShadowPass::CascadedShadowPass(ResourceRegistry &resources, Config cfg)
    : m_resources(resources), m_cfg(std::move(cfg))
{
    if (!m_cfg.camera || m_cfg.resolution == 0 || m_cfg.shadowDistance <= 0.0f ||
        m_cfg.splitLambda < 0.0f || m_cfg.splitLambda > 1.0f || m_cfg.depthPadding < 0.0f)
        throw std::invalid_argument("CascadedShadowPass: invalid configuration");
    resources.registerPersistentImageArray(m_cfg.shadowImage, VK_FORMAT_D32_SFLOAT,
                                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                           {m_cfg.resolution, m_cfg.resolution}, CascadedShadowGpuData::maxLayers,
                                           VK_IMAGE_ASPECT_DEPTH_BIT);
    resources.registerDynamicBuffer(m_cfg.paramsBuffer, sizeof(CascadedShadowGpuData),
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    const CascadedShadowGpuData disabled{};
    resources.updateBuffer(m_cfg.paramsBuffer, &disabled, sizeof(disabled));
}

void CascadedShadowPass::setViewportExtent(VkExtent2D extent)
{
    if (extent.height != 0)
        m_aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
}

CascadedShadowGpuData CascadedShadowPass::shadowData() const
{
    CascadedShadowGpuData result{};
    if (!m_cfg.camera->hasComponent<Camera>() || !m_cfg.camera->hasComponent<TransformComponent>())
        return result;
    const Camera &camera = m_cfg.camera->getComponent<Camera>();
    const Transform &cameraTransform = m_cfg.camera->getComponent<TransformComponent>().transform();
    const float nearPlane = std::max(camera.nearPlane, 0.001f);
    const float farPlane = std::max(nearPlane + 0.001f, std::min(camera.farPlane, m_cfg.shadowDistance));
    std::array<float, CascadedShadowGpuData::cascadeCount + 1> splits{};
    splits[0] = nearPlane;
    for (uint32_t cascade = 1; cascade <= CascadedShadowGpuData::cascadeCount; ++cascade)
    {
        const float ratio = static_cast<float>(cascade) / CascadedShadowGpuData::cascadeCount;
        const float logarithmic = nearPlane * std::pow(farPlane / nearPlane, ratio);
        const float uniform = nearPlane + (farPlane - nearPlane) * ratio;
        splits[cascade] = glm::mix(uniform, logarithmic, m_cfg.splitLambda);
        result.splitDepths[cascade - 1] = splits[cascade];
    }

    for (SceneObject *object : m_cfg.lightObjects)
    {
        if (!object || !object->scene().contains(object->id()) || !object->hasComponent<Light>() ||
            !std::holds_alternative<DirectionalLight>(object->getComponent<Light>().light))
            continue;
        const glm::vec3 lightDirection = object->getComponent<TransformComponent>().transform().forward();
        const auto &directional = std::get<DirectionalLight>(object->getComponent<Light>().light);
        const float tanAngularRadius = std::tan(glm::radians(directional.angularRadiusDegrees));
        const uint32_t lightIndex = result.header.x++;
        for (uint32_t cascade = 0; cascade < CascadedShadowGpuData::cascadeCount; ++cascade)
        {
            const auto corners = frustumSliceCorners(camera, cameraTransform, m_aspect,
                                                     splits[cascade], splits[cascade + 1]);
            glm::vec3 center(0.0f);
            for (const glm::vec3 &corner : corners) center += corner;
            center /= static_cast<float>(corners.size());
            float radius = 0.0f;
            for (const glm::vec3 &corner : corners) radius = std::max(radius, glm::length(corner - center));
            radius = std::ceil(radius * 16.0f) / 16.0f;
            const glm::vec3 worldUp = std::abs(glm::dot(lightDirection, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.99f
                                        ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
            // Snap the centre in a light frame that does not depend on the centre itself, otherwise the
            // texel grid follows the frustum centroid and slides as the camera rotates. The quantised
            // radius above keeps texelSize constant between frames, which is what makes the lattice stable.
            const glm::mat4 lightRotation = glm::lookAt(glm::vec3(0.0f), lightDirection, worldUp);
            const float texelSize = (2.0f * radius) / static_cast<float>(m_cfg.resolution);
            glm::vec3 centerLight = glm::vec3(lightRotation * glm::vec4(center, 1.0f));
            centerLight.x = std::floor(centerLight.x / texelSize) * texelSize;
            centerLight.y = std::floor(centerLight.y / texelSize) * texelSize;
            center = glm::vec3(glm::transpose(lightRotation) * glm::vec4(centerLight, 1.0f));

            const glm::vec3 eye = center - lightDirection * (radius + m_cfg.depthPadding);
            const glm::mat4 view = glm::lookAt(eye, center, worldUp);
            glm::mat4 projection = glm::ortho(-radius, radius, -radius, radius,
                                              0.01f, 2.0f * radius + 2.0f * m_cfg.depthPadding);
            projection[1][1] *= -1.0f;
            glm::mat4 clip(1.0f);
            clip[2][2] = 0.5f;
            clip[3][2] = 0.5f;
            const uint32_t layer = lightIndex * CascadedShadowGpuData::cascadeCount + cascade;
            result.lightViewProj[layer] = clip * projection * view;
            // An orthographic depth buffer is linear, so a stored-depth difference between blocker
            // and receiver scales straight to a world distance by the cascade's depth range. The
            // penumbra that distance subtends converts to UV through the cascade's own ortho width,
            // which is what keeps the filter the same world size across cascades of different extent.
            const float depthRange = 2.0f * radius + 2.0f * m_cfg.depthPadding - 0.01f;
            result.pcss[layer].x = depthRange * tanAngularRadius / (2.0f * radius);
        }
        if (result.header.x == CascadedShadowGpuData::maxLights) break;
    }
    result.header.y = CascadedShadowGpuData::cascadeCount;
    return result;
}

void CascadedShadowPass::setSceneGeometry(VertexBufferUploadResult vertices, IndexBufferUploadResult indices,
                                          std::vector<const TransformComponent *> transforms,
                                          std::vector<SceneObject *> objects, std::vector<SkinDrawInfo> skins)
{
    m_cfg.geometry.vertexBufferUploadResult = std::move(vertices);
    m_cfg.geometry.indexBufferUploadResult = std::move(indices);
    m_cfg.geometry.meshTransforms = std::move(transforms);
    m_cfg.geometry.meshObjects = std::move(objects);
    m_cfg.geometry.skinDrawInfos = std::move(skins);
}

void CascadedShadowPass::build(FrameGraph &fg, const GpuMeshLayout &layout) const
{
    auto pass = fg.addPass("cascadedShadow").type(PassType::Geometry).vertexLayout(layout);
    for (const auto &[binding, name] : m_cfg.geometry.vertexBufferResourceNames)
        pass.vertexBuffer(binding, fg.buffer(name));
    pass.indexBuffer(fg.buffer(m_cfg.geometry.indexBufferResourceName))
        .vertShader((paths::shaderDir / "cascaded_shadow.vert.spv").string())
        .fragShader((paths::shaderDir / "cascaded_shadow.frag.spv").string())
        .cull(VK_CULL_MODE_BACK_BIT).depthBias(1.25f, 1.75f)
        .renderingLayers(CascadedShadowGpuData::maxLayers)
        .pushConstantSize(sizeof(ShadowPC), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .uniformBuffer(0, fg.buffer(m_cfg.paramsBuffer), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledImageArray(1, fg.image(m_cfg.geometry.diffuseTextureArrayResourceName), m_cfg.geometry.materialCount,
                           VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(5, fg.buffer(m_cfg.geometry.faceGroupBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(6, fg.buffer(m_cfg.geometry.materialBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(7, fg.buffer(m_cfg.geometry.skinInfluenceEntriesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(8, fg.buffer(m_cfg.geometry.skinInfluenceOffsetsBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(9, fg.buffer(m_cfg.geometry.skinPositionIndicesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(10, fg.buffer(m_cfg.geometry.skinJointMatricesBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .depthAttachment(fg.image(m_cfg.shadowImage), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR,
                         {.depthStencil = {1.0f, 0}}, ExtentSpec::absolute(m_cfg.resolution, m_cfg.resolution))
        .execute([this](PassContext &ctx) {
            const CascadedShadowGpuData data = shadowData();
            m_resources.updateBuffer(m_cfg.paramsBuffer, &data, sizeof(data));
            const uint32_t layerCount = data.header.x * data.header.y;
            if (layerCount == 0) return;
            const auto &g = m_cfg.geometry;
            for (size_t i = 0; i < g.vertexBufferUploadResult.singleMeshResults.size(); ++i)
            {
                if (!g.meshObjects[i]->scene().contains(g.meshObjects[i]->id())) continue;
                const auto &mesh = g.vertexBufferUploadResult.singleMeshResults[i];
                const auto &range = g.indexBufferUploadResult.singleMeshResults[i];
                const auto *transform = g.meshTransforms[i];
                const auto &skin = g.skinDrawInfos[i];
                const ShadowPC pc{.model = transform ? transform->worldMatrix() : glm::mat4(1.0f),
                                  .primitiveIdOffset = range.firstIndex / 3,
                                  .paletteOffset = skin.paletteOffset,
                                  .skinEnabled = skin.skinEnabled ? 1u : 0u};
                ctx.cmd().pushConstants(ctx.pipelineLayout(),
                                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
                ctx.cmd().drawIndexed(range.indexCount, layerCount, range.firstIndex, mesh.vertexOffset, 0);
            }
        });
}
} // namespace lr
