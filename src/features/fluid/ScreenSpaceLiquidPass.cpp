#include "features/fluid/ScreenSpaceLiquidPass.hpp"

#include "core/Paths.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace lr
{
namespace
{

constexpr const char *kDepthRaw       = "fluidDepthRaw";
constexpr const char *kDepthTemp      = "fluidDepthTemp";
constexpr const char *kDepthSmooth    = "fluidDepthSmooth";
constexpr const char *kSurfaceDepth   = "fluidSurfaceDepth";
constexpr const char *kNormal         = "fluidNormal";
constexpr const char *kThickness      = "fluidThickness";
constexpr const char *kParams         = "fluidLiquidParams";

struct alignas(16) LiquidParams
{
    glm::vec4 absorptionAndRefraction;
    glm::vec4 filterAndMaterial;
    glm::vec4 tintAndFresnel;
    glm::ivec4 filterRadius;
};

static_assert(sizeof(LiquidParams) == 64);

LiquidParams buildParams(const ScreenSpaceLiquidPass::Config &config)
{
    const glm::vec3 tint = glm::clamp(config.liquidColor, glm::vec3(0.0f), glm::vec3(1.0f));
    const float spatialSigma = std::max(config.spatialSigma, 0.01f);
    // A Gaussian is effectively zero past three sigma. Sizing the kernel from the sigma stops a
    // wide blur from being silently truncated by a small radius, which would leave a boxy
    // footprint instead of the intended falloff. The configured radius remains the upper bound so
    // the cost per pixel stays predictable.
    const int sigmaRadius = static_cast<int>(std::ceil(3.0f * spatialSigma));
    const int radius = std::clamp(std::min(config.filterRadius, sigmaRadius), 1, 12);
    return {
        .absorptionAndRefraction = glm::vec4(
            screen_space_liquid::absorptionCoefficient(tint, config.absorptionScale),
            std::max(config.refractionStrength, 0.0f)),
        .filterAndMaterial = glm::vec4(spatialSigma,
                                      std::max(config.depthSigma, 0.0001f),
                                      std::max(config.thicknessScale, 0.0f),
                                      std::max(config.indexOfRefraction, 1.0f)),
        .tintAndFresnel = glm::vec4(tint, screen_space_liquid::fresnelF0(
                                              std::max(config.indexOfRefraction, 1.0f))),
        .filterRadius = glm::ivec4(radius, 0, 0, 0),
    };
}

} // namespace

ScreenSpaceLiquidPass::ScreenSpaceLiquidPass(Config config, ResourceRegistry &resources)
    : m_config(std::move(config)), m_resources(&resources)
{
    if (m_config.cameraBufferResourceName.empty())
    {
        throw std::invalid_argument("ScreenSpaceLiquidPass requires a camera buffer resource");
    }
    if (m_config.maxParticles == 0)
    {
        throw std::invalid_argument("ScreenSpaceLiquidPass maxParticles must be greater than zero");
    }

    resources.registerDynamicBuffer(m_config.particleBufferResourceName,
                                    sizeof(ParticleGpuData) * m_config.maxParticles,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    resources.registerDynamicBuffer(kParams, sizeof(LiquidParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    const VkImageUsageFlags storageAndSampled = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    resources.registerImage(kDepthTemp, VK_FORMAT_R32_SFLOAT, storageAndSampled);
    resources.registerImage(kDepthSmooth, VK_FORMAT_R32_SFLOAT, storageAndSampled);
    resources.registerImage(kNormal, VK_FORMAT_R16G16B16A16_SFLOAT, storageAndSampled);
    updateParams();
}

void ScreenSpaceLiquidPass::setParticles(std::span<const ParticleGpuData> particles)
{
    m_particleCount = particle_billboard::checkedParticleCount(particles.size(), m_config.maxParticles);
    if (!particles.empty())
    {
        m_resources->updateBuffer(m_config.particleBufferResourceName, particles.data(),
                                  sizeof(ParticleGpuData) * particles.size());
    }
}

void ScreenSpaceLiquidPass::setLiquidColor(const glm::vec3 &color)
{
    if (glm::all(glm::equal(m_config.liquidColor, color)))
    {
        return;
    }
    m_config.liquidColor = color;
    updateParams();
}

void ScreenSpaceLiquidPass::updateParams()
{
    const LiquidParams params = buildParams(m_config);
    m_resources->updateBuffer(kParams, &params, sizeof(params));
}

void ScreenSpaceLiquidPass::build(FrameGraph &frameGraph)
{
    VkClearValue zero{};
    VkClearValue farDepth{};
    farDepth.depthStencil = {1.0f, 0};

    // The depth attachment resolves overlapping impostors exactly; the R32 target keeps the
    // corresponding positive view-space distance for filtering and reconstruction.
    frameGraph.addPass("fluid_nearest_depth")
        .type(PassType::Geometry)
        .topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .vertShader((paths::shaderDir / "fluid_surface.vert.spv").string())
        .fragShader((paths::shaderDir / "fluid_depth.frag.spv").string())
        .uniformBuffer(0, frameGraph.buffer(m_config.cameraBufferResourceName),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(1, frameGraph.buffer(m_config.particleBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledDepth(2, frameGraph.image("gbufferDepth"), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::depthFetch())
        .blend(BlendMode::Opaque)
        .depth(true, true, VK_COMPARE_OP_LESS)
        .cull(VK_CULL_MODE_NONE)
        .colorAttachment(frameGraph.image(kDepthRaw), VK_FORMAT_R32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR, zero)
        .depthAttachment(frameGraph.image(kSurfaceDepth), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR,
                         farDepth)
        .execute([this](PassContext &context) {
            if (m_particleCount != 0)
            {
                context.cmd().draw(6, m_particleCount);
            }
        });

    const auto addBlur = [&](const char *name, ImageHandle input, ImageHandle output, glm::ivec2 direction) {
        frameGraph.addPass(name)
            .type(PassType::Compute)
            .computeShader((paths::shaderDir / "fluid_bilateral.comp.spv").string())
            .pushConstantSize(sizeof(glm::ivec2), VK_SHADER_STAGE_COMPUTE_BIT)
            .sampledImage(0, input, VK_SHADER_STAGE_COMPUTE_BIT, SamplerDesc::clampToEdge())
            .uniformBuffer(1, frameGraph.buffer(kParams), VK_SHADER_STAGE_COMPUTE_BIT)
            .storageImageWrite(2, output, VK_SHADER_STAGE_COMPUTE_BIT)
            .execute([output, direction](PassContext &context) {
                const VkExtent2D extent = context.extent(output);
                context.cmd().pushConstants(context.pipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, direction);
                context.cmd().dispatch(screen_space_liquid::dispatchSize(extent.width),
                                       screen_space_liquid::dispatchSize(extent.height), 1);
            });
    };
    addBlur("fluid_bilateral_horizontal", frameGraph.image(kDepthRaw), frameGraph.image(kDepthTemp), {1, 0});
    addBlur("fluid_bilateral_vertical", frameGraph.image(kDepthTemp), frameGraph.image(kDepthSmooth), {0, 1});

    frameGraph.addPass("fluid_normal_reconstruction")
        .type(PassType::Compute)
        .computeShader((paths::shaderDir / "fluid_normal.comp.spv").string())
        .uniformBuffer(0, frameGraph.buffer(m_config.cameraBufferResourceName), VK_SHADER_STAGE_COMPUTE_BIT)
        .sampledImage(1, frameGraph.image(kDepthSmooth), VK_SHADER_STAGE_COMPUTE_BIT,
                      SamplerDesc::clampToEdge())
        .storageImageWrite(2, frameGraph.image(kNormal), VK_SHADER_STAGE_COMPUTE_BIT)
        .execute([normal = frameGraph.image(kNormal)](PassContext &context) {
            const VkExtent2D extent = context.extent(normal);
            context.cmd().dispatch(screen_space_liquid::dispatchSize(extent.width),
                                   screen_space_liquid::dispatchSize(extent.height), 1);
        });

    // Every visible sphere contributes its analytic chord length. Additive blending turns the
    // overlapping cloud into an optical path length without sorting particles.
    frameGraph.addPass("fluid_thickness")
        .type(PassType::Geometry)
        .topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .vertShader((paths::shaderDir / "fluid_surface.vert.spv").string())
        .fragShader((paths::shaderDir / "fluid_thickness.frag.spv").string())
        .uniformBuffer(0, frameGraph.buffer(m_config.cameraBufferResourceName),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(1, frameGraph.buffer(m_config.particleBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledDepth(2, frameGraph.image("gbufferDepth"), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::depthFetch())
        .uniformBuffer(3, frameGraph.buffer(kParams), VK_SHADER_STAGE_FRAGMENT_BIT)
        .blend(BlendMode::Additive)
        .depth(false, false)
        .cull(VK_CULL_MODE_NONE)
        .colorAttachment(frameGraph.image(kThickness), VK_FORMAT_R32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR,
                         zero)
        .execute([this](PassContext &context) {
            if (m_particleCount != 0)
            {
                context.cmd().draw(6, m_particleCount);
            }
        });

    frameGraph.addPass("fluid_composite")
        .type(PassType::Fullscreen)
        .vertShader((paths::shaderDir / "fullscreen.vert.spv").string())
        .fragShader((paths::shaderDir / "fluid_composite.frag.spv").string())
        .uniformBuffer(0, frameGraph.buffer(m_config.cameraBufferResourceName), VK_SHADER_STAGE_FRAGMENT_BIT)
        .sampledImage(1, frameGraph.image(kDepthSmooth), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::clampToEdge())
        .sampledImage(2, frameGraph.image(kNormal), VK_SHADER_STAGE_FRAGMENT_BIT, SamplerDesc::clampToEdge())
        .sampledImage(3, frameGraph.image(kThickness), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::clampToEdge())
        .sampledImage(4, frameGraph.image("pbr"), VK_SHADER_STAGE_FRAGMENT_BIT, SamplerDesc::clampToEdge())
        .sampledDepth(5, frameGraph.image("gbufferDepth"), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::depthFetch())
        .sampledImage(6, frameGraph.image("ibl_env"), VK_SHADER_STAGE_FRAGMENT_BIT,
                      SamplerDesc::clampToEdge())
        .uniformBuffer(7, frameGraph.buffer(kParams), VK_SHADER_STAGE_FRAGMENT_BIT)
        .colorAttachment(frameGraph.image(compositeImageName()), VK_FORMAT_R16G16B16A16_SFLOAT,
                         VK_ATTACHMENT_LOAD_OP_CLEAR, zero)
        .execute([](PassContext &context) { context.cmd().draw(3); });
}

} // namespace lr
