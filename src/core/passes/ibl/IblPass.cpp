#include "IblPass.hpp"

#include "core/Paths.hpp"
#include "core/utility/ImageLoader.hpp"

#include <glm/vec4.hpp>

#include <spdlog/spdlog.h>

#include <stdexcept>
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
    float    roughness;
    uint32_t sampleCount;
};

struct BrdfLutPC
{
    uint32_t sampleCount;
};

LoadedHdrImage loadConfiguredHdri(const IBLPass::Config &config)
{
    if (!config.hdriData.empty()) return loadHdrFromMemory(config.hdriData, config.hdriPath.string());
    if (!config.hdriPath.empty()) return loadHdrFromFile(config.hdriPath);
    return LoadedHdrImage::singlePixel(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
}

} // namespace

IBLPass::IBLPass(Config cfg) : m_cfg(std::move(cfg)) {}

void IBLPass::uploadResources(ResourceRegistry &resources) const
{
    LoadedHdrImage hdri;
    if (m_cfg.hdriPath.empty() && m_cfg.hdriData.empty())
    {
        spdlog::info("IBLPass: no HDRI configured; using a black environment");
        hdri = LoadedHdrImage::singlePixel(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    } else
    {
        try
        {
            hdri = loadConfiguredHdri(m_cfg);
        } catch (const std::exception &e)
        {
            spdlog::warn("IBLPass: failed to load HDRI '{}': {}. Using a black environment.", m_cfg.hdriPath.string(),
                         e.what());
            hdri = LoadedHdrImage::singlePixel(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        }
    }

    if (hdri.empty())
    {
        throw std::runtime_error("IBLPass: failed to allocate fallback HDR pixel");
    }

    resources.uploadImage("hdri", hdri.pixels, hdri.width, hdri.height, LoadedHdrImage::format);

    // This LUT is used in the PBR shader, but we register it here since it is most relevant...
    // It is integrated on the GPU rather than loaded from an 8-bit PNG: the bias term is added
    // straight into the specular result, so 1/255 steps in it read as rings on smooth surfaces.
    // TRANSFER_SRC is not needed to render; it lets the table be copied back and diffed against a
    // reference integrator (see tests/vulkan/BrdfLutGpuSmoke.cpp).
    resources.registerPersistentImage("ibl_brdf_lut", VK_FORMAT_R32G32_SFLOAT,
                                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                      {m_cfg.lutRes, m_cfg.lutRes});

    resources.registerCubemap("ibl_env", VK_FORMAT_R16G16B16A16_SFLOAT, m_cfg.envRes, 1,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    resources.registerCubemap("ibl_irradiance", VK_FORMAT_R16G16B16A16_SFLOAT, m_cfg.irrRes, 1,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    resources.registerCubemap("ibl_prefiltered", VK_FORMAT_R16G16B16A16_SFLOAT, m_cfg.pfRes, m_cfg.pfMips,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
}

void IBLPass::replaceHdriResource(ResourceRegistry &resources) const
{
    LoadedHdrImage hdri = loadConfiguredHdri(m_cfg);

    if (hdri.empty())
    {
        throw std::runtime_error("IBLPass: failed to allocate fallback HDR pixel");
    }

    resources.replaceUploadedImage("hdri", hdri.pixels, hdri.width, hdri.height, LoadedHdrImage::format);
}

void IBLPass::build(FrameGraph &fg) const
{
    const uint32_t envRes = m_cfg.envRes;
    const uint32_t irrRes = m_cfg.irrRes;
    const uint32_t pfRes  = m_cfg.pfRes;
    const uint32_t pfMips = m_cfg.pfMips;
    const uint32_t lutRes = m_cfg.lutRes;

    // Independent of the HDRI — it only depends on the BRDF — but it costs a fraction of a
    // millisecond once at startup, so it rides along with the rest of the preprocessing graph.
    const BrdfLutPC lutPC = {.sampleCount = 1024};

    fg.addPass("ibl_brdf_lut")
        .type(PassType::Compute)
        .computeShader((paths::shaderDir / "brdflut.comp.spv").string())
        .pushConstantSize(sizeof(BrdfLutPC))
        .storageImageWrite(0, fg.image("ibl_brdf_lut"), VK_SHADER_STAGE_COMPUTE_BIT)
        .execute([lutPC, lutRes](PassContext &ctx) {
            ctx.cmd().pushConstants(ctx.pipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, lutPC);
            ctx.cmd().dispatch((lutRes + 15) / 16, (lutRes + 15) / 16, 1);
        });

    fg.addPass("ibl_hdri_to_cube")
        .type(PassType::Compute)
        .computeShader((paths::shaderDir / "hdritocubemap.comp.spv").string())
        .sampledImage(0, fg.image("hdri"), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageImageWrite(1, fg.image("ibl_env"), VK_SHADER_STAGE_COMPUTE_BIT)
        .execute([envRes](PassContext &ctx) {
            ctx.cmd().dispatch((envRes + 15) / 16, (envRes + 15) / 16, 6);
        });

    const IrradiancePC irrPC = {
        .nsamples    = {64, 32, 0, 0},
        .sampleDelta = {2.0f * kPi / 64.0f, 0.5f * kPi / 32.0f, 0.0f, 0.0f},
    };

    fg.addPass("ibl_irradiance")
        .type(PassType::Compute)
        .computeShader((paths::shaderDir / "irradiance.comp.spv").string())
        .pushConstantSize(sizeof(IrradiancePC))
        .sampledImage(0, fg.image("ibl_env"), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageImageWrite(1, fg.image("ibl_irradiance"), VK_SHADER_STAGE_COMPUTE_BIT)
        .execute([irrPC, irrRes](PassContext &ctx) {
            ctx.cmd().pushConstants(ctx.pipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, irrPC);
            ctx.cmd().dispatch((irrRes + 15) / 16, (irrRes + 15) / 16, 6);
        });

    for (uint32_t mip = 0; mip < pfMips; ++mip)
    {
        const float    roughness = static_cast<float>(mip) / static_cast<float>(pfMips - 1);
        const uint32_t mipSize   = pfRes >> mip;

        const PrefilterPC pfPC = {
            .roughness   = roughness,
            .sampleCount = 4096,
        };

        fg.addPass("ibl_prefilter_mip" + std::to_string(mip))
            .type(PassType::Compute)
            .computeShader((paths::shaderDir / "prefilter.comp.spv").string())
            .pushConstantSize(sizeof(PrefilterPC))
            .sampledImage(0, fg.image("ibl_env"), VK_SHADER_STAGE_COMPUTE_BIT)
            .storageImageWrite(1, ImageView::mip(fg.image("ibl_prefiltered"), mip), VK_SHADER_STAGE_COMPUTE_BIT)
            .execute([pfPC, mipSize](PassContext &ctx) {
                ctx.cmd().pushConstants(ctx.pipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, pfPC);
                ctx.cmd().dispatch((mipSize + 15) / 16, (mipSize + 15) / 16, 6);
            });
    }
}

} // namespace lr
