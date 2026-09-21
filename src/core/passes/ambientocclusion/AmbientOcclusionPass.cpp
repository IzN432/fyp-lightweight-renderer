#include "core/passes/ambientocclusion/AmbientOcclusionPass.hpp"

#include "core/Paths.hpp"

#include <glm/glm.hpp>

#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace lr
{

namespace
{

struct AOParamUBO
{
    glm::vec4 sphereRadius; // r, r^2, 1/r, 0
    int       numSteps;
    int       numDirs;
    float     tanAngleBias;
    float     aoScalar;
};

constexpr uint32_t kGroupSize = 16;

uint32_t dispatchSize(uint32_t pixels) { return (pixels + kGroupSize - 1) / kGroupSize; }

} // namespace

AmbientOcclusionPass::AmbientOcclusionPass(Config cfg) : m_cfg(std::move(cfg)) {}

void AmbientOcclusionPass::uploadResources(ResourceRegistry &resources) const
{
    // hbao_ao — pre-registered so the blur pass can write it as a storage image. hbao_raw is the
    // unblurred HBAO output, read by the blur pass (see build()).
    const VkImageUsageFlags storageAndSampled = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    resources.registerImage("hbao_ao", VK_FORMAT_R32_SFLOAT, storageAndSampled);
    resources.registerImage("hbao_raw", VK_FORMAT_R32_SFLOAT, storageAndSampled);

    // 4x4 tile of random directions, tiled over the screen — seeded for reproducibility
    constexpr uint32_t                    kNoiseSize = 4;
    std::vector<glm::vec4>                dirs(kNoiseSize * kNoiseSize);
    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (glm::vec4 &dir : dirs)
    {
        const float angle = dist(rng) * 2.0f * 3.14159265f;
        dir               = glm::vec4(std::cos(angle), std::sin(angle), 0.0f, 1.0f); // z reserved for jitter
    }
    resources.uploadImage("hbao_directions", dirs.data(), kNoiseSize, kNoiseSize, VK_FORMAT_R32G32B32A32_SFLOAT);

    const float      r = m_cfg.sphereRadius;
    const AOParamUBO params{
        .sphereRadius = glm::vec4(r, r * r, 1.0f / r, 0.0f),
        .numSteps     = m_cfg.numSteps,
        .numDirs      = m_cfg.numDirs,
        .tanAngleBias = m_cfg.tanAngleBias,
        .aoScalar     = m_cfg.aoScalar,
    };
    resources.uploadBuffer("hbao_params", &params, sizeof(params), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

void AmbientOcclusionPass::build(FrameGraph &fg) const
{
    const ImageHandle raw = fg.image("hbao_raw");
    const ImageHandle ao  = fg.image("hbao_ao");

    // Full-res HBAO compute, unblurred. Deinterleave/interleave are disconnected for now (shaders
    // kept in the tree); they will be reintroduced incrementally.
    fg.addPass("hbao_ao")
        .type(PassType::Compute)
        .computeShader((paths::shaderDir / "hbao.comp.spv").string())
        .uniformBuffer(0, fg.buffer(m_cfg.cameraBufferResourceName), VK_SHADER_STAGE_COMPUTE_BIT)
        .sampledDepth(1, fg.image("gbufferDepth"), VK_SHADER_STAGE_COMPUTE_BIT)
        .sampledImage(2, fg.image("gbufferNormal"), VK_SHADER_STAGE_COMPUTE_BIT)
        .sampledImage(3, fg.image("hbao_directions"), VK_SHADER_STAGE_COMPUTE_BIT)
        .uniformBuffer(4, fg.buffer("hbao_params"), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageImageWrite(5, raw, VK_SHADER_STAGE_COMPUTE_BIT)
        .execute([raw](PassContext &ctx) {
            const VkExtent2D extent = ctx.extent(raw);
            ctx.cmd().dispatch(dispatchSize(extent.width), dispatchSize(extent.height), 1);
        });

    // Joint (cross) bilateral blur — a single true 2D 5x5 pass. (We tried splitting this into
    // separate X/Y 1D passes; it's a well-known approximation for a non-separable filter, and it
    // showed up here as a visible dotted artifact along diagonal depth edges, so we reverted to
    // the exact 2D version.)
    fg.addPass("hbao_blur")
        .type(PassType::Compute)
        .computeShader((paths::shaderDir / "hbao_blur.comp.spv").string())
        .sampledImage(0, raw, VK_SHADER_STAGE_COMPUTE_BIT)
        .sampledDepth(1, fg.image("gbufferDepth"), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageImageWrite(2, ao, VK_SHADER_STAGE_COMPUTE_BIT)
        .execute([ao](PassContext &ctx) {
            const VkExtent2D extent = ctx.extent(ao);
            ctx.cmd().dispatch(dispatchSize(extent.width), dispatchSize(extent.height), 1);
        });
}

} // namespace lr
