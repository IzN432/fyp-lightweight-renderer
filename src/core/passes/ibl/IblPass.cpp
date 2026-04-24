#include "IblPass.hpp"

#include "core/utility/ImageLoader.hpp"

#include <glm/vec4.hpp>

#include <string>

namespace lr
{

namespace
{

constexpr float kPi = 3.14159265359f;

struct IrradiancePC
{
    glm::ivec4 nsamples;
    glm::vec4  sampleDelta;
};

struct PrefilterPC
{
    glm::vec4  roughnessPadded;
    glm::ivec4 nsamples;
    glm::vec4  sampleDelta;
};

}  // namespace

IBLPass::IBLPass(Config cfg)
    : m_cfg(std::move(cfg))
{
}

void IBLPass::uploadResources(ResourceRegistry &resources) const
{
    {
        LoadedHdrImage hdri = loadHdrFromFile(m_cfg.hdriPath);
        resources.uploadImage("hdri", hdri.pixels, hdri.width, hdri.height,
                              LoadedHdrImage::format);
    }

    resources.registerCubemap("ibl_env",
        VK_FORMAT_R16G16B16A16_SFLOAT, m_cfg.envRes, 1,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    resources.registerCubemap("ibl_irradiance",
        VK_FORMAT_R16G16B16A16_SFLOAT, m_cfg.irrRes, 1,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    resources.registerCubemap("ibl_prefiltered",
        VK_FORMAT_R16G16B16A16_SFLOAT, m_cfg.pfRes, m_cfg.pfMips,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
}

void IBLPass::preprocess(FrameGraph &fg) const
{
    const auto    &shaderDir = m_cfg.shaderDir;
    const uint32_t envRes    = m_cfg.envRes;
    const uint32_t irrRes    = m_cfg.irrRes;
    const uint32_t pfRes     = m_cfg.pfRes;
    const uint32_t pfMips    = m_cfg.pfMips;

    fg.addPass("ibl_hdri_to_cube")
        .type(PassType::Compute)
        .computeShader((shaderDir / "hdritocubemap.comp.spv").string())
        .bind({
            {.resourceName = "hdri",    .binding = 0,
             .type   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             .stages = VK_SHADER_STAGE_COMPUTE_BIT},
            {.resourceName = "ibl_env", .binding = 1,
             .type   = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
             .access = BindingAccess::Write},
        })
        .execute([envRes](CommandBuffer &cmd, VkPipelineLayout) {
            cmd.dispatch((envRes + 15) / 16, (envRes + 15) / 16, 6);
        });

    const IrradiancePC irrPC = {
        .nsamples    = {64, 32, 0, 0},
        .sampleDelta = {2.0f * kPi / 64.0f, 0.5f * kPi / 32.0f, 0.0f, 0.0f},
    };

    fg.addPass("ibl_irradiance")
        .type(PassType::Compute)
        .computeShader((shaderDir / "irradiance.comp.spv").string())
        .pushConstantSize(sizeof(IrradiancePC))
        .bind({
            {.resourceName = "ibl_env",        .binding = 0,
             .type   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             .stages = VK_SHADER_STAGE_COMPUTE_BIT},
            {.resourceName = "ibl_irradiance", .binding = 1,
             .type   = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
             .access = BindingAccess::Write},
        })
        .execute([irrPC, irrRes](CommandBuffer &cmd, VkPipelineLayout layout) {
            cmd.pushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT, irrPC);
            cmd.dispatch((irrRes + 15) / 16, (irrRes + 15) / 16, 6);
        });

    for (uint32_t mip = 0; mip < pfMips; ++mip)
    {
        const float    roughness = static_cast<float>(mip) / static_cast<float>(pfMips - 1);
        const uint32_t mipSize   = pfRes >> mip;

        const PrefilterPC pfPC = {
            .roughnessPadded = {roughness, 0.0f, 0.0f, 0.0f},
            .nsamples        = {32, 32, 32 * 32, 0},
            .sampleDelta     = {0.1f, 0.1f, static_cast<float>(envRes), 0.0f},
        };

        fg.addPass("ibl_prefilter_mip" + std::to_string(mip))
            .type(PassType::Compute)
            .computeShader((shaderDir / "prefilter.comp.spv").string())
            .pushConstantSize(sizeof(PrefilterPC))
            .bind({
                {.resourceName = "ibl_env",        .binding = 0,
                 .type   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                 .stages = VK_SHADER_STAGE_COMPUTE_BIT},
                {.resourceName = "ibl_prefiltered", .binding = 1,
                 .type     = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                 .stages   = VK_SHADER_STAGE_COMPUTE_BIT,
                 .access   = BindingAccess::Write,
                 .mipLevel = mip},
            })
            .execute([pfPC, mipSize](CommandBuffer &cmd, VkPipelineLayout layout) {
                cmd.pushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT, pfPC);
                cmd.dispatch((mipSize + 15) / 16, (mipSize + 15) / 16, 6);
            });
    }

    fg.executeOnce();
}

}  // namespace lr
