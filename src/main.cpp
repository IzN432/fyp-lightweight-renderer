#include "core/app/Viewer.hpp"
#include "core/loaders/GltfLoader.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Mesh.hpp"
#include "core/upload/CameraUploader.hpp"
#include "core/upload/MeshUploader.hpp"
#include "core/utility/ImageLoader.hpp"

#include <imgui.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec4.hpp>
#include <spdlog/spdlog.h>

#include <cmath>
#include <filesystem>
#include <stdexcept>

int main()
try
{
    spdlog::set_level(spdlog::level::debug);

    lr::Viewer viewer({.title = "lr"});

    // -------------------------------------------------------------------------
    // IBL preprocessing  (runs once before the frame loop)
    // -------------------------------------------------------------------------

    namespace fs = std::filesystem;
    const fs::path shaderDir = LR_SHADER_DIR;

    // Resolutions — lower values are faster to precompute during development
    constexpr uint32_t ENV_RES  = 2048;   // equirect → cubemap
    constexpr uint32_t IRR_RES  = 32;    // irradiance (diffuse IBL)
    constexpr uint32_t PF_RES   = 512;   // prefiltered specular, base mip
    constexpr uint32_t PF_MIPS  = 10;     // 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1

    // Source HDRI — change this path to your .hdr file
    const fs::path hdriPath = "C:\\Users\\seani\\Downloads\\cedar_bridge_sunset_2_4k.hdr";
    {
        lr::LoadedHdrImage hdri = lr::loadHdrFromFile(hdriPath);
        viewer.resources().uploadImage("hdri", hdri.pixels,
                                       hdri.width, hdri.height,
                                       lr::LoadedHdrImage::format);
    }

    // Output cubemaps registered here; the preprocess passes write into them
    viewer.resources().registerCubemap("ibl_env",
        VK_FORMAT_R16G16B16A16_SFLOAT, ENV_RES, 1,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    viewer.resources().registerCubemap("ibl_irradiance",
        VK_FORMAT_R16G16B16A16_SFLOAT, IRR_RES, 1,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    viewer.resources().registerCubemap("ibl_prefiltered",
        VK_FORMAT_R16G16B16A16_SFLOAT, PF_RES, PF_MIPS,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    // Pass 1: equirectangular HDRI → environment cubemap
    viewer.frameGraph().addPass("ibl_hdri_to_cube")
        .type(lr::PassType::Compute)
        .computeShader((shaderDir / "hdritocubemap.comp.spv").string())
        .bind({
            {.resourceName = "hdri",    .binding = 0,
             .type   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             .stages = VK_SHADER_STAGE_COMPUTE_BIT},
            {.resourceName = "ibl_env", .binding = 1,
             .type   = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
             .access = lr::BindingAccess::Write},
        })
        .execute([](lr::CommandBuffer &cmd, VkPipelineLayout) {
            cmd.dispatch((ENV_RES + 15) / 16, (ENV_RES + 15) / 16, 6);
        });

    // Pass 2: environment cubemap → irradiance (diffuse IBL)
    struct IrradiancePC { glm::ivec4 nsamples; glm::vec4 sampleDelta; };
    constexpr float PI_f = 3.14159265359f;
    const IrradiancePC irrPC = {
        .nsamples    = {64, 32, 0, 0},
        .sampleDelta = {2.0f * PI_f / 64.0f, 0.5f * PI_f / 32.0f, 0.0f, 0.0f},
    };

    viewer.frameGraph().addPass("ibl_irradiance")
        .type(lr::PassType::Compute)
        .computeShader((shaderDir / "irradiance.comp.spv").string())
        .pushConstantSize(sizeof(IrradiancePC))
        .bind({
            {.resourceName = "ibl_env",       .binding = 0,
             .type   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             .stages = VK_SHADER_STAGE_COMPUTE_BIT},
            {.resourceName = "ibl_irradiance", .binding = 1,
             .type   = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
             .access = lr::BindingAccess::Write},
        })
        .execute([irrPC](lr::CommandBuffer &cmd, VkPipelineLayout layout) {
            cmd.pushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT, irrPC);
            cmd.dispatch((IRR_RES + 15) / 16, (IRR_RES + 15) / 16, 6);
        });

    // Pass 3: environment cubemap → prefiltered specular (one pass per mip / roughness level)
    struct PrefilterPC { glm::vec4 roughnessPadded; glm::ivec4 nsamples; glm::vec4 sampleDelta; };

    for (uint32_t mip = 0; mip < PF_MIPS; ++mip)
    {
        const float    roughness = static_cast<float>(mip) / static_cast<float>(PF_MIPS - 1);
        const uint32_t mipSize   = PF_RES >> mip;

        const PrefilterPC pfPC = {
            .roughnessPadded = {roughness, 0.0f, 0.0f, 0.0f},
            .nsamples        = {32, 32, 32 * 32, 0},
            .sampleDelta     = {0.1f, 0.1f, static_cast<float>(ENV_RES), 0.0f},
        };

        viewer.frameGraph().addPass("ibl_prefilter_mip" + std::to_string(mip))
            .type(lr::PassType::Compute)
            .computeShader((shaderDir / "prefilter.comp.spv").string())
            .pushConstantSize(sizeof(PrefilterPC))
            .bind({
                {.resourceName = "ibl_env",        .binding = 0,
                 .type   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                 .stages = VK_SHADER_STAGE_COMPUTE_BIT},
                {.resourceName = "ibl_prefiltered", .binding = 1,
                 .type     = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                 .stages   = VK_SHADER_STAGE_COMPUTE_BIT,
                 .access   = lr::BindingAccess::Write,
                 .mipLevel = mip},
            })
            .execute([pfPC, mipSize](lr::CommandBuffer &cmd, VkPipelineLayout layout) {
                cmd.pushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT, pfPC);
                cmd.dispatch((mipSize + 15) / 16, (mipSize + 15) / 16, 6);
            });
    }

    // Compile all preprocessing passes, submit once, discard — registry keeps the results
    viewer.frameGraph().executeOnce();

    // -------------------------------------------------------------------------
    // Scene setup
    // -------------------------------------------------------------------------

    const glm::vec3 cameraTarget(0.0f, 0.0f, 0.0f);
    const float     cameraOrbitRadius = 5.0f;

    lr::Camera camera;
    camera.transform.position = glm::vec3(0.0f, 0.0f, cameraOrbitRadius);
    {
        const glm::mat4 view = glm::lookAt(camera.transform.position,
                                           cameraTarget,
                                           glm::vec3(0.0f, 1.0f, 0.0f));
        camera.transform.rotation = glm::conjugate(glm::quat_cast(view));
    }

    lr::MeshLayout layout;
    layout.addPerVertexAttr<glm::vec3>("normal");
    layout.addPerVertexAttr<glm::vec2>("uv");
    layout.enableFaceGroups();

    const std::filesystem::path meshPath = "D:\\Blender\\Projects\\helmet.glb";
    lr::GltfLoader gltfLoader;
    lr::MeshSequence meshSequence = gltfLoader.load(meshPath, layout);
    if (meshSequence.empty())
        throw std::runtime_error("GltfLoader returned empty sequence for '" + meshPath.string() + "'");

    lr::GpuMeshLayout gpuLayout(layout);
    gpuLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT);
    gpuLayout.map("normal", 0, 1, VK_FORMAT_R32G32B32_SFLOAT);
    gpuLayout.map("uv",     0, 2, VK_FORMAT_R32G32_SFLOAT);

    // -------------------------------------------------------------------------
    // Resource uploads
    // -------------------------------------------------------------------------

    lr::CameraUploader cameraUploader(viewer.resources());

    lr::MeshUploader meshUploader(viewer.resources());
    const lr::MeshUploadResult mesh = meshUploader.upload(
        meshSequence.frames.front(),
        gpuLayout,
        meshSequence.materials,
        "mesh");

    // -------------------------------------------------------------------------
    // Frame graph passes
    // -------------------------------------------------------------------------

    viewer.frameGraph().addPass("geometry")
        .type(lr::PassType::Geometry)
        .vertexLayout(gpuLayout)
        .vertexBuffer(0, mesh.vertexBufferNames.at(0))
        .indexBuffer(mesh.indexBufferName)
        .indexCount(mesh.indexCount)
        .vertShader((shaderDir / "geometry.vert.spv").string())
        .fragShader((shaderDir / "geometry.frag.spv").string())
        .bind({
            {
                .resourceName = lr::CameraUploader::kBufferName,
                .binding      = 0,
                .type         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .stages       = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName    = mesh.diffuseTextureArrayName,
                .binding         = 1,
                .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = mesh.materialTextureCount,
                .stages          = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName    = mesh.specularTextureArrayName,
                .binding         = 2,
                .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = mesh.materialTextureCount,
                .stages          = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName    = mesh.normalTextureArrayName,
                .binding         = 3,
                .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = mesh.materialTextureCount,
                .stages          = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName    = mesh.roughnessTextureArrayName,
                .binding         = 4,
                .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = mesh.materialTextureCount,
                .stages          = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName    = mesh.metallicTextureArrayName,
                .binding         = 5,
                .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = mesh.materialTextureCount,
                .stages          = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName    = mesh.emissiveTextureArrayName,
                .binding         = 6,
                .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = mesh.materialTextureCount,
                .stages          = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName = mesh.faceGroupIndexBufferName,
                .binding      = 7,
                .type         = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
            }
        })
        .writes({
            {.name = "gbufferAlbedo", .format = VK_FORMAT_R16G16B16A16_SFLOAT},
            {.name = "gbufferNormal", .format = VK_FORMAT_R16G16_SFLOAT},
            {.name = "gbufferMaterial", .format = VK_FORMAT_R8G8B8A8_UNORM},
            {.name = "swapchain",     .format = viewer.frameGraph().resources().getImage("swapchain")->format},
            {.name = "gbufferDepth",  .format = VK_FORMAT_D32_SFLOAT,
             .clearValue = {.depthStencil = {1.0f, 0}}}
        })
        .execute([indexCount = mesh.indexCount](lr::CommandBuffer &cmd, VkPipelineLayout) {
            cmd.drawIndexed(indexCount);
        });

    // Final pass: fullscreen cubemap background
    viewer.frameGraph().addPass("final")
        .type(lr::PassType::Fullscreen)
        .vertShader((shaderDir / "fullscreen.vert.spv").string())
        .fragShader((shaderDir / "final.frag.spv").string())
        .bind({
            {
                .resourceName = lr::CameraUploader::kBufferName,
                .binding      = 0,
                .type         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName = "ibl_env",
                .binding      = 1,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .resourceName = "gbufferDepth",
                .binding      = 2,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "gbufferAlbedo",
                .binding      = 3,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "gbufferNormal",
                .binding      = 4,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .resourceName = "gbufferMaterial",
                .binding      = 5,
                .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stages       = VK_SHADER_STAGE_FRAGMENT_BIT,
                .imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            }
        })
        .writes({
            {.name = "swapchain", .format = viewer.frameGraph().resources().getImage("swapchain")->format}
        })
        .execute([](lr::CommandBuffer &cmd, VkPipelineLayout) {
            cmd.draw(3); 
        });

    // -------------------------------------------------------------------------
    // Per-frame callbacks
    // -------------------------------------------------------------------------

    viewer.onGui([]() {
        ImGui::ShowDemoWindow();
    });

    float orbitAngle = 0.0f;

    viewer.onUpdate([&](float dt, VkExtent2D extent) {
        const float aspect = (extent.height == 0)
            ? 1.0f
            : static_cast<float>(extent.width) / static_cast<float>(extent.height);

        orbitAngle += dt * 0.5f;
        camera.transform.position = glm::vec3(
            std::sin(orbitAngle) * cameraOrbitRadius,
            0.0f,
            std::cos(orbitAngle) * cameraOrbitRadius);
        {
            const glm::mat4 view = glm::lookAt(camera.transform.position,
                                               cameraTarget,
                                               glm::vec3(0.0f, 1.0f, 0.0f));
            camera.transform.rotation = glm::conjugate(glm::quat_cast(view));
        }

        cameraUploader.upload(camera, aspect);
    });

    viewer.run();
    return 0;
}
catch (const std::exception &e)
{
    spdlog::error("Fatal: {}", e.what());
    return 1;
}
