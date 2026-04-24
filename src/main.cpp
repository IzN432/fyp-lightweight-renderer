#include "core/app/Viewer.hpp"
#include "core/loaders/GltfLoader.hpp"
#include "core/passes/final/FinalPass.hpp"
#include "core/passes/geometry/GeometryPass.hpp"
#include "core/passes/ibl/IblPass.hpp"
#include "core/passes/pbr/PBRPass.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Mesh.hpp"
#include "core/upload/CameraUploader.hpp"
#include "core/upload/MaterialUploader.hpp"
#include "core/upload/MeshUploader.hpp"

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

    namespace fs = std::filesystem;
    const fs::path shaderDir = LR_SHADER_DIR;

    // -------------------------------------------------------------------------
    // IBL preprocessing  (runs once before the frame loop)
    // -------------------------------------------------------------------------

    lr::IBLPass iblPass({
        .hdriPath  = "C:\\Users\\seani\\Downloads\\cedar_bridge_sunset_2_4k.hdr",
        .shaderDir = shaderDir,
    });
    iblPass.uploadResources(viewer.resources());
    iblPass.preprocess(viewer.frameGraph());

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

    const fs::path meshPath = "C:\\Users\\seani\\Documents\\monke.glb";

    lr::GltfLoader gltfLoader;
    lr::GltfLoaderConfig config{
        .normalAttributeName = "normal",
        .uvAttributeName = "uv",
        .diffuseTextureName = "baseColorTexture",
        .normalTextureName = "normalTexture",
        .metallicRoughnessTextureName = "metallicRoughnessTexture",
        .emissiveTextureName = "emissiveTexture",
        .baseDiffuseName = "baseDiffuse",
        .baseRoughnessName = "baseRoughness",
        .baseMetallicName = "baseMetallic",
        .baseEmissiveName = "baseEmissive",
    };
    auto [sequence, layout, materials] = gltfLoader.load(meshPath, config);

    if (sequence.empty())
        throw std::runtime_error("GltfLoader returned empty sequence for '" + meshPath.string() + "'");

        
    // -------------------------------------------------------------------------
    // Resource uploads
    // -------------------------------------------------------------------------
        
    lr::CameraUploader cameraUploader(viewer.resources());

    lr::GpuMeshLayout gpuMeshLayout(layout);
    gpuMeshLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT);
    gpuMeshLayout.map(config.normalAttributeName, 0, 1, VK_FORMAT_R32G32B32_SFLOAT);
    gpuMeshLayout.map(config.uvAttributeName, 0, 2, VK_FORMAT_R32G32_SFLOAT);

    lr::MeshUploader meshUploader(viewer.resources());
    const lr::MeshUploadResult mesh = meshUploader.upload(
        sequence.frames.front(),
        gpuMeshLayout,
        "mesh");

    // This matches the expected layout in geometry.frag
    lr::GpuMaterialLayout gpuMaterialLayout;
    gpuMaterialLayout
        .setStride(48)
        .addScalar(config.baseDiffuseName, 0, sizeof(glm::vec4))
        .addScalar(config.baseEmissiveName, 16, sizeof(glm::vec3))
        .addScalar(config.baseRoughnessName, 32, sizeof(float))
        .addScalar(config.baseMetallicName, 36, sizeof(float))
        .addTexture(config.diffuseTextureName)
        .addTexture(config.normalTextureName)
        .addTexture(config.metallicRoughnessTextureName)
        .addTexture(config.emissiveTextureName);

    lr::MaterialUploader materialUploader(viewer.resources());
    const lr::MaterialUploadResult material = materialUploader.upload(
        materials,
        gpuMaterialLayout,
        "material");
    
    // -------------------------------------------------------------------------
    // Frame graph passes
    // -------------------------------------------------------------------------

    const VkFormat swapchainFormat =
        viewer.frameGraph().resources().getImage("swapchain")->format;

    lr::GeometryPass geometryPass({
        .shaderDir = shaderDir,

        .vertexAttributeBufferResourceName = mesh.vertexBufferNames.at(0),
        .indexBufferResourceName = mesh.indexBufferName,
        .indexCount = mesh.indexCount,

        .faceGroupBufferResourceName = mesh.faceGroupBufferName,

        .diffuseTextureArrayResourceName = material.textureNameMap.at(config.diffuseTextureName),
        .normalTextureArrayResourceName = material.textureNameMap.at(config.normalTextureName),
        .metallicRoughnessTextureArrayResourceName = material.textureNameMap.at(config.metallicRoughnessTextureName),
        .emissiveTextureArrayResourceName = material.textureNameMap.at(config.emissiveTextureName),
        .materialBufferResourceName = material.materialInfoBufferName,
        
        .materialCount = static_cast<uint32_t>(materials.size()),

        .swapchainFormat = swapchainFormat,
    });
    geometryPass.build(viewer.frameGraph(), gpuMeshLayout);

    lr::PbrPass pbrPass({
        .shaderDir = shaderDir,
        .swapchainFormat = swapchainFormat,
    });
    pbrPass.build(viewer.frameGraph());
    
    lr::FinalPass finalPass({
        .shaderDir = shaderDir,
        .swapchainFormat = swapchainFormat,
    });
    finalPass.build(viewer.frameGraph());

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
