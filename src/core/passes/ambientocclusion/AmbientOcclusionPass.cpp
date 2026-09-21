#include "core/passes/ambientocclusion/AmbientOcclusionPass.hpp"

#include "core/Paths.hpp"

#include <glm/glm.hpp>

#include <string>

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

AOParamUBO buildParams(const AmbientOcclusionPass::Config &cfg)
{
    const float r = cfg.sphereRadius;
    return AOParamUBO{
        .sphereRadius = glm::vec4(r, r * r, 1.0f / r, 0.0f),
        .numSteps     = cfg.numSteps,
        .numDirs      = cfg.numDirs,
        .tanAngleBias = cfg.tanAngleBias,
        .aoScalar     = cfg.aoScalar,
    };
}

} // namespace

AmbientOcclusionPass::AmbientOcclusionPass(Config cfg) : m_cfg(std::move(cfg)) {}

void AmbientOcclusionPass::uploadResources(ResourceRegistry &resources) const
{
    // hbao_ao — pre-registered so the blur pass can write it as a storage image. hbao_raw is the
    // unblurred HBAO output, read by the blur pass (see build()).
    const VkImageUsageFlags storageAndSampled = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    resources.registerImage("hbao_ao", VK_FORMAT_R32_SFLOAT, storageAndSampled);
    resources.registerImage("hbao_raw", VK_FORMAT_R32_SFLOAT, storageAndSampled);

    // Per-pixel rotation and step jitter are now a procedural hash of pos computed directly in
    // hbao.comp — no direction texture to seed here (see its doc comment for why: a small repeating
    // tile produces a visible periodic pattern that an edge-preserving blur can't remove).

    // Dynamic (not uploadBuffer's static GPU_ONLY) so updateParams() can push edits from a GUI
    // every frame without needing a destroy/recreate.
    resources.registerDynamicBuffer("hbao_params", sizeof(AOParamUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    const AOParamUBO params = buildParams(m_cfg);
    resources.updateBuffer("hbao_params", &params, sizeof(params));
}

void AmbientOcclusionPass::updateParams(ResourceRegistry &resources) const
{
    const AOParamUBO params = buildParams(m_cfg);
    resources.updateBuffer("hbao_params", &params, sizeof(params));
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
        .uniformBuffer(3, fg.buffer("hbao_params"), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageImageWrite(4, raw, VK_SHADER_STAGE_COMPUTE_BIT)
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
