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
    glm::vec4 sphereRadius;  // r, r^2, 1/r, 0
    int       numSteps;
    int       numDirs;
    float     tanAngleBias;
    float     aoScalar;
};

constexpr uint32_t kGroupSize = 16;

uint32_t dispatchSize(uint32_t pixels)
{
    return (pixels + kGroupSize - 1) / kGroupSize;
}

}  // namespace

AmbientOcclusionPass::AmbientOcclusionPass(Config cfg)
    : m_cfg(std::move(cfg))
{
}

void AmbientOcclusionPass::uploadResources(ResourceRegistry &resources) const
{
    VkExtent2D full     = resources.getExtent();
    uint32_t   halfW    = std::max(1u, full.width  / 2);
    uint32_t   halfH    = std::max(1u, full.height / 2);
    VkExtent2D halfSize = {halfW, halfH};

    // Half-res depth submaps (one per 2x2 quadrant) — persistent so they don't
    // resize to full-res on swapchain resize (same behaviour as the reference renderer)
    const VkImageUsageFlags storageAndSampled =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    for (int i = 0; i < 4; i++) {
        resources.registerPersistentImage("hbao_depth_"  + std::to_string(i),
                                          VK_FORMAT_R32_SFLOAT,  storageAndSampled, halfSize);
        resources.registerPersistentImage("hbao_normal_" + std::to_string(i),
                                          VK_FORMAT_R32G32_SFLOAT, storageAndSampled, halfSize);
        resources.registerPersistentImage("hbao_ao_sub_" + std::to_string(i),
                                          VK_FORMAT_R32_SFLOAT,  storageAndSampled, halfSize);
    }

    // Full-res intermediate — written by interleave compute, read by blur compute pass
    resources.registerImage("hbao_raw", VK_FORMAT_R32_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    // hbao_ao — pre-registered so the blur compute pass can write it as a storage image
    resources.registerImage("hbao_ao", VK_FORMAT_R32_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    // Random sampling directions — seeded for reproducibility
    const int numDirs = m_cfg.numDirs;
    std::vector<glm::vec4> dirs(numDirs);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < numDirs; i++) {
        const float angle = dist(rng) * 2.0f * 3.14159265f;
        dirs[i] = glm::vec4(std::cos(angle), std::sin(angle), dist(rng), 1.0f);
    }
    resources.uploadImage("hbao_directions", dirs.data(),
                          static_cast<uint32_t>(numDirs), 1,
                          VK_FORMAT_R32G32B32A32_SFLOAT);

    const float r = m_cfg.sphereRadius;
    const AOParamUBO params{
        .sphereRadius = glm::vec4(r, r * r, 1.0f / r, 0.0f),
        .numSteps     = m_cfg.numSteps,
        .numDirs      = m_cfg.numDirs,
        .tanAngleBias = m_cfg.tanAngleBias,
        .aoScalar     = m_cfg.aoScalar,
    };
    resources.uploadBuffer("hbao_params", &params, sizeof(params),
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

void AmbientOcclusionPass::build(FrameGraph &fg) const
{
    // Capture dispatch sizes from the registry at build time.
    // Persistent half-res images don't resize, so these stay correct.
    const VkExtent2D full  = fg.resources().getExtent();
    const uint32_t   halfW = std::max(1u, full.width  / 2);
    const uint32_t   halfH = std::max(1u, full.height / 2);
    const uint32_t   fullW = full.width;
    const uint32_t   fullH = full.height;

    // ------------------------------------------------------------------ //
    // 1. Deinterleave depth + normal → 4 half-res submaps each
    // ------------------------------------------------------------------ //
    fg.addPass("hbao_deinterleave")
        .type(PassType::Compute)
        .computeShader((paths::shaderDir / "deinterleave.comp.spv").string())
        .bind({
            {
                .resourceName = "gbufferDepth",
                .binding      = 0,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_COMPUTE_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "gbufferNormal",
                .binding      = 1,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_COMPUTE_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {.resourceName="hbao_depth_0", .binding=2,.type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,.stages=VK_SHADER_STAGE_COMPUTE_BIT,.access=BindingAccess::Write},
            {.resourceName="hbao_depth_1", .binding=3,.type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,.stages=VK_SHADER_STAGE_COMPUTE_BIT,.access=BindingAccess::Write},
            {.resourceName="hbao_depth_2", .binding=4,.type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,.stages=VK_SHADER_STAGE_COMPUTE_BIT,.access=BindingAccess::Write},
            {.resourceName="hbao_depth_3", .binding=5,.type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,.stages=VK_SHADER_STAGE_COMPUTE_BIT,.access=BindingAccess::Write},
            {.resourceName="hbao_normal_0",.binding=6,.type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,.stages=VK_SHADER_STAGE_COMPUTE_BIT,.access=BindingAccess::Write},
            {.resourceName="hbao_normal_1",.binding=7,.type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,.stages=VK_SHADER_STAGE_COMPUTE_BIT,.access=BindingAccess::Write},
            {.resourceName="hbao_normal_2",.binding=8,.type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,.stages=VK_SHADER_STAGE_COMPUTE_BIT,.access=BindingAccess::Write},
            {.resourceName="hbao_normal_3",.binding=9,.type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,.stages=VK_SHADER_STAGE_COMPUTE_BIT,.access=BindingAccess::Write},
        })
        .execute([halfW, halfH](CommandBuffer &cmd, VkPipelineLayout) {
            cmd.dispatch(dispatchSize(halfW), dispatchSize(halfH), 1);
        });

    // ------------------------------------------------------------------ //
    // 3. HBAO compute — one pass per quadrant, each at half resolution
    // ------------------------------------------------------------------ //
    for (int i = 0; i < 4; i++) {
        const int passId = i;
        fg.addPass("hbao_ao_" + std::to_string(i))
            .type(PassType::Compute)
            .computeShader((paths::shaderDir / "hbao.comp.spv").string())
            .pushConstantSize(sizeof(int), VK_SHADER_STAGE_COMPUTE_BIT)
            .bind({
                {
                    .resourceName = m_cfg.cameraBufferResourceName,
                    .binding      = 0,
                    .type         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .stages       = VK_SHADER_STAGE_COMPUTE_BIT,
                },
                {
                    .resourceName = "hbao_depth_"  + std::to_string(i),
                    .binding      = 1,
                    .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .stages       = VK_SHADER_STAGE_COMPUTE_BIT,
                    .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                },
                {
                    .resourceName = "hbao_normal_" + std::to_string(i),
                    .binding      = 2,
                    .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .stages       = VK_SHADER_STAGE_COMPUTE_BIT,
                    .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                },
                {
                    .resourceName = "hbao_directions",
                    .binding      = 3,
                    .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .stages       = VK_SHADER_STAGE_COMPUTE_BIT,
                    .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                },
                {
                    .resourceName = "hbao_params",
                    .binding      = 4,
                    .type         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .stages       = VK_SHADER_STAGE_COMPUTE_BIT,
                },
                {
                    .resourceName = "hbao_ao_sub_" + std::to_string(i),
                    .binding      = 5,
                    .type         = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .stages       = VK_SHADER_STAGE_COMPUTE_BIT,
                    .access       = BindingAccess::Write,
                },
            })
            .execute([passId, halfW, halfH](CommandBuffer &cmd, VkPipelineLayout layout) {
                cmd.pushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT, passId);
                cmd.dispatch(dispatchSize(halfW), dispatchSize(halfH), 1);
            });
    }

    // ------------------------------------------------------------------ //
    // 4. Interleave — reassemble 4 half-res AO submaps into full-res
    // ------------------------------------------------------------------ //
    fg.addPass("hbao_interleave")
        .type(PassType::Compute)
        .computeShader((paths::shaderDir / "interleave.comp.spv").string())
        .bind({
            {.resourceName="hbao_ao_sub_0",.binding=0,.type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.stages=VK_SHADER_STAGE_COMPUTE_BIT,.imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {.resourceName="hbao_ao_sub_1",.binding=1,.type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.stages=VK_SHADER_STAGE_COMPUTE_BIT,.imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {.resourceName="hbao_ao_sub_2",.binding=2,.type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.stages=VK_SHADER_STAGE_COMPUTE_BIT,.imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {.resourceName="hbao_ao_sub_3",.binding=3,.type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.stages=VK_SHADER_STAGE_COMPUTE_BIT,.imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {.resourceName="hbao_raw",.binding=4,.type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,.stages=VK_SHADER_STAGE_COMPUTE_BIT,.access=BindingAccess::Write},
        })
        .execute([fullW, fullH](CommandBuffer &cmd, VkPipelineLayout) {
            cmd.dispatch(dispatchSize(fullW), dispatchSize(fullH), 1);
        });

    // ------------------------------------------------------------------ //
    // 5. Bilateral blur — compute with shared-memory tile, output is hbao_ao
    // ------------------------------------------------------------------ //
    fg.addPass("hbao_blur")
        .type(PassType::Compute)
        .computeShader((paths::shaderDir / "hbao_blur.comp.spv").string())
        .bind({
            {
                .resourceName = "hbao_raw",
                .binding      = 0,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_COMPUTE_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "gbufferDepth",
                .binding      = 1,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_COMPUTE_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "hbao_ao",
                .binding      = 2,
                .type         = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .stages       = VK_SHADER_STAGE_COMPUTE_BIT,
                .access       = BindingAccess::Write,
            },
        })
        .execute([fullW, fullH](CommandBuffer &cmd, VkPipelineLayout) {
            cmd.dispatch(dispatchSize(fullW), dispatchSize(fullH), 1);
        });
}

}  // namespace lr
