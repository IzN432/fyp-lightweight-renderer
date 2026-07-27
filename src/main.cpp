#include "core/app/Viewer.hpp"
#include "core/loaders/GltfLoader.hpp"
#include "core/overlay/OverlayMesh.hpp"
#include "core/passes/final/FinalPass.hpp"
#include "core/passes/geometry/GeometryPass.hpp"
#include "core/passes/ibl/IblPass.hpp"
#include "core/passes/pbr/PbrPass.hpp"
#include "core/passes/ambientocclusion/AmbientOcclusionPass.hpp"
#include "core/passes/overlaygeometry/OverlayGeometryPass.hpp"
#include "core/passes/overlaypoints/OverlayPointsPass.hpp"
#include "core/framegraph/ImageReadback.hpp"

#include "core/scene/Camera.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/StaticMesh.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/upload/CameraUploader.hpp"
#include "core/upload/LightUploader.hpp"
#include "core/upload/MaterialUploader.hpp"
#include "core/upload/MeshUploader.hpp"
#include "core/editor/camera/SphericalCameraController.hpp"
#include "core/editor/gizmo/GizmoManager.hpp"
#include "core/editor/gizmo/translate/TranslateArrowGizmo.hpp"
#include "core/editor/gizmo/translate/TranslateBoxGizmo.hpp"
#include "core/editor/selection/BoxSelectionTool.hpp"
#include "core/editor/selection/SelectionManager.hpp"
#include "core/editor/VertexManager.hpp"

#include <imgui.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec4.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

int main()
try
{
    spdlog::set_level(spdlog::level::debug);

    lr::Viewer viewer({.title = "lr"});

    namespace fs = std::filesystem;

    // -------------------------------------------------------------------------
    // IBL preprocessing  (runs once before the frame loop)
    // -------------------------------------------------------------------------

    lr::IBLPass iblPass({
        .hdriPath  = "C:\\Users\\seani\\Downloads\\cedar_bridge_sunset_2_4k.hdr",
        .envRes    = 2048,
        .irrRes    = 32,
        .pfRes     = 2048,
        .pfMips    = 8
    });
    iblPass.uploadResources(viewer.resources());
    iblPass.preprocess(viewer.frameGraph());

    // -------------------------------------------------------------------------
    // Scene setup
    // -------------------------------------------------------------------------

    std::vector<std::unique_ptr<lr::SceneObject>> sceneObjects;

    lr::SceneObject* camera = sceneObjects.emplace_back(std::make_unique<lr::SceneObject>()).get();
    camera->addComponent<lr::Camera>();
    camera->addComponent<lr::Transform>();
    camera->name = "Main Camera";

    // LIGHT
    lr::DirectionalLight light;
    light.color = glm::vec3(1.0f, 1.0f, 1.0f);
    light.intensity = 1.0f;
    
    lr::SceneObject* editorLight = sceneObjects.emplace_back(std::make_unique<lr::SceneObject>()).get();
    editorLight->addComponent<lr::Transform>();
    editorLight->addComponent<lr::Light>(light);
    editorLight->name = "Directional Light";

    lr::LightUploader lightUploader(viewer.resources());

    // MESH
    const fs::path meshPath = "D:\\FYP\\lion_head_4k.blend\\lion_head_4k.glb";

    lr::GltfLoader gltfLoader;
    lr::GltfLoaderConfig config{
        .normalAttributeName = "normal",
        .tangentAttributeName = "tangent",
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
    auto [sequence, materials] = gltfLoader.load(meshPath, config);

    if (sequence.empty())
        throw std::runtime_error("GltfLoader returned empty sequence for '" + meshPath.string() + "'");

    lr::SceneObject* meshObject = sceneObjects.emplace_back(std::make_unique<lr::SceneObject>()).get();
    meshObject->addComponent<lr::Transform>();
    
    {
        lr::Mesh &m = sequence.frames.front();
        std::vector<glm::vec3> colors(m.vertexCount(), glm::vec3(1.0f, 0.0f, 1.0f));
        m.setPerVertexArray("color", std::span<const glm::vec3>(colors));
    }
    auto &staticMesh = meshObject->addComponent<lr::StaticMesh>(sequence.frames.front(), materials);
    meshObject->name = "Mesh Object";

    // -------------------------------------------------------------------------
    // Resource uploads
    // -------------------------------------------------------------------------
        
    lr::CameraUploader cameraUploader(viewer.resources());

    float aspect = 1600.0f / 900.0f;
    std::function<void()> updateCameraUpload = [&camera, &cameraUploader, &aspect]() {
        cameraUploader.upload(*camera, aspect);
    };
    camera->getComponent<lr::Camera>().addChangeListener(updateCameraUpload);
    camera->getComponent<lr::Transform>().addChangeListener(updateCameraUpload);

    std::function<void()> updateLightList = [&sceneObjects, &lightUploader]() {
        std::vector<lr::SceneObject*> sceneLights;
        for (const auto &object : sceneObjects) {
            if (object->hasComponent<lr::Light>()) {
                sceneLights.push_back(object.get());
            }
        }
        lightUploader.upload(sceneLights);
    };

    std::vector<lr::SceneObject*> sceneLights;
    for (const auto &sceneObject : sceneObjects) {
        if (sceneObject->hasComponent<lr::Light>()) {
            sceneLights.push_back(sceneObject.get());
            sceneObject->getComponent<lr::Light>().addChangeListener(updateLightList);
            sceneObject->getComponent<lr::Transform>().addChangeListener(updateLightList);
        }
    }

    lightUploader.upload(sceneLights);
    
    // Upload the main scene geometry
    lr::MeshUploader meshUploader(viewer.resources());
    const std::string mainMeshPositionBufferName  = "meshPositionBuffer";
    const std::string mainMeshVertexBufferName    = "meshVertexBuffer";
    const std::string mainMeshColorBufferName     = "meshColorBuffer";
    const std::string mainMeshIndexBufferName     = "meshIndexBuffer";
    const std::string mainMeshFaceGroupBufferName = "meshFaceGroupBuffer";

    const lr::VertexBufferUploadResult meshPositions = meshUploader.uploadVertexBuffer(
        { &staticMesh.mesh() },
        { .vertexBufferName = mainMeshPositionBufferName,
          .includePosition  = true });
    meshUploader.uploadVertexBuffer(
        { &staticMesh.mesh() },
        { .vertexBufferName       = mainMeshVertexBufferName,
          .vertexAttributeNames   = { config.normalAttributeName, config.tangentAttributeName, config.uvAttributeName } });
    // Color buffer is dynamic so selection highlights can be updated each frame.
    std::vector<glm::vec3> pointColors(staticMesh.mesh().vertexCount(), glm::vec3(1.0f, 0.0f, 1.0f));
    viewer.resources().registerDynamicBuffer(
        mainMeshColorBufferName,
        pointColors.size() * sizeof(glm::vec3),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    viewer.resources().updateBuffer(
        mainMeshColorBufferName,
        pointColors.data(),
        pointColors.size() * sizeof(glm::vec3));
    const lr::IndexBufferUploadResult indexBuffer = meshUploader.uploadIndexBuffer(
        { &staticMesh.mesh() },
        { .indexBufferName = mainMeshIndexBufferName });
    meshUploader.uploadFaceGroupBuffer(
        { &staticMesh.mesh() },
        { .faceGroupBufferName = mainMeshFaceGroupBufferName });

    // This matches the expected layout in geometry.frag
    lr::GpuMaterialLayout gpuMaterialLayout;
    gpuMaterialLayout
        .setStride(48)
        .addScalar(config.baseDiffuseName, 0, sizeof(glm::vec4))
        .addScalar(config.baseEmissiveName, 16, sizeof(glm::vec3))
        .addScalar(config.baseRoughnessName, 32, sizeof(float))
        .addScalar(config.baseMetallicName, 36, sizeof(float))
        .addTexture(config.diffuseTextureName,            VK_FORMAT_R8G8B8A8_SRGB)
        .addTexture(config.normalTextureName,             VK_FORMAT_R8G8B8A8_UNORM)
        .addTexture(config.metallicRoughnessTextureName,  VK_FORMAT_R8G8B8A8_UNORM)
        .addTexture(config.emissiveTextureName,           VK_FORMAT_R8G8B8A8_SRGB);

    lr::MaterialUploader materialUploader(viewer.resources());
    const lr::MaterialUploadResult material = materialUploader.upload(
        staticMesh.materials(),
        gpuMaterialLayout,
        "material");
    
    staticMesh.addChangeListener([&staticMesh, &materialUploader, &gpuMaterialLayout, &material]() {
        materialUploader.update(
            staticMesh.materials(),
            gpuMaterialLayout,
            material);
    });

    // -------------------------------------------------------------------------
    // Frame graph passes
    // -------------------------------------------------------------------------

    const VkFormat swapchainFormat =
        viewer.frameGraph().resources().getImage("swapchain")->format;

    lr::GeometryPass geometryPass({
        .cameraBufferResourceName  = cameraUploader.bufferName(),
        .vertexBufferResourceNames = { {0, mainMeshPositionBufferName}, {1, mainMeshVertexBufferName} },
        .vertexBufferUploadResult  = meshPositions,
        .indexBufferUploadResult   = indexBuffer,
        .indexBufferResourceName = mainMeshIndexBufferName,
        .faceGroupBufferResourceName = mainMeshFaceGroupBufferName,
        .diffuseTextureArrayResourceName = material.textureNameMap.at(config.diffuseTextureName),
        .normalTextureArrayResourceName = material.textureNameMap.at(config.normalTextureName),
        .metallicRoughnessTextureArrayResourceName = material.textureNameMap.at(config.metallicRoughnessTextureName),
        .emissiveTextureArrayResourceName = material.textureNameMap.at(config.emissiveTextureName),
        .materialBufferResourceName = material.materialInfoBufferName,

        .materialCount = static_cast<uint32_t>(staticMesh.materials().size()),
    });
    lr::GpuMeshLayout gpuMeshLayout(staticMesh.mesh().layout());

    gpuMeshLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT);
    gpuMeshLayout.map(config.normalAttributeName,  1, 1, VK_FORMAT_R32G32B32_SFLOAT);
    gpuMeshLayout.map(config.tangentAttributeName, 1, 2, VK_FORMAT_R32G32B32A32_SFLOAT);
    gpuMeshLayout.map(config.uvAttributeName,      1, 3, VK_FORMAT_R32G32_SFLOAT);

    geometryPass.build(viewer.frameGraph(), gpuMeshLayout);

    lr::AmbientOcclusionPass aoPass({
        .cameraBufferResourceName = cameraUploader.bufferName(),
    });
    aoPass.uploadResources(viewer.resources());
    aoPass.build(viewer.frameGraph());

    lr::PbrPass pbrPass({
        .cameraBufferResourceName = cameraUploader.bufferName(),
        .lightBufferResourceName = lightUploader.bufferName(),
        .numLights = lightUploader.numLights(),
        .pfMips = 8,
    });
    pbrPass.build(viewer.frameGraph());
    
    lr::OverlayGeometryPass overlayGeometryPass({
        .cameraBufferResourceName = cameraUploader.bufferName(),
    });
    overlayGeometryPass.uploadResources(viewer.resources());
    overlayGeometryPass.build(viewer.frameGraph());
    overlayGeometryPass.setInstances({});

    lr::GpuMeshLayout pointsMeshLayout(staticMesh.mesh().layout());
    pointsMeshLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT);
    pointsMeshLayout.map("color", 1, 1, VK_FORMAT_R32G32B32_SFLOAT);

    lr::OverlayPointsPass overlayPointsPass({
        .cameraBufferResourceName   = cameraUploader.bufferName(),
        .positionBufferResourceName = mainMeshPositionBufferName,
        .colorBufferResourceName    = mainMeshColorBufferName,
        .positionBufferUploadResult = meshPositions,
        .vertexCounts               = { staticMesh.mesh().vertexCount() },
    });
    overlayPointsPass.build(viewer.frameGraph(), pointsMeshLayout);

    lr::FinalPass finalPass({
        .cameraBufferResourceName = cameraUploader.bufferName(),
        .swapchainFormat = swapchainFormat,
    });
    finalPass.build(viewer.frameGraph());

    // -------------------------------------------------------------------------
    // Editor state — vertex picking, selection and gizmo managers
    // -------------------------------------------------------------------------

    bool displayPoints = true;

    // Gizmo hover — reads the picking image from the previous frame
    lr::ImageReadback gizmoReadback(viewer.context(), viewer.allocator());

    lr::VertexManager vertexManager(staticMesh.mesh().positions);
    vertexManager.registerUpdateCallback([&]() {
        meshUploader.updateVertexBuffer(
            { &staticMesh.mesh() },
            { .vertexBufferName = mainMeshPositionBufferName,
              .includePosition  = true });
    });

    lr::SelectionManager selectionManager(staticMesh.mesh().positions, viewer.input());
    selectionManager.setSelectTool(std::make_unique<lr::BoxSelectionTool>(viewer.input(), *camera));
    selectionManager.registerSelectionChangedCallback([&]() {
        std::fill(pointColors.begin(), pointColors.end(), glm::vec3(1.0f, 0.0f, 1.0f));
        for (uint32_t idx : selectionManager.getSelectedIndices())
            pointColors[idx] = glm::vec3(1.0f, 0.8f, 0.0f);  // orange = selected
        viewer.resources().updateBuffer(
            mainMeshColorBufferName,
            pointColors.data(),
            pointColors.size() * sizeof(glm::vec3));
    });

    lr::GizmoManager gizmoManager(overlayGeometryPass, viewer.input());
    const std::vector<int> translateGizmoIds = {
        gizmoManager.addGizmo(std::make_unique<lr::TranslateArrowGizmo>(
            lr::TranslateArrowGizmoAxis::X, *camera, viewer.input(), vertexManager, selectionManager)),
        gizmoManager.addGizmo(std::make_unique<lr::TranslateArrowGizmo>(
            lr::TranslateArrowGizmoAxis::Y, *camera, viewer.input(), vertexManager, selectionManager)),
        gizmoManager.addGizmo(std::make_unique<lr::TranslateArrowGizmo>(
            lr::TranslateArrowGizmoAxis::Z, *camera, viewer.input(), vertexManager, selectionManager)),
        gizmoManager.addGizmo(std::make_unique<lr::TranslateBoxGizmo>(
            *camera, viewer.input(), vertexManager, selectionManager)),
    };
    for (int id : translateGizmoIds)
        gizmoManager.hideGizmo(id);

    // Single combined LMB handler: gizmos get first refusal on a click (so
    // dragging an arrow doesn't simultaneously start a box-select), and
    // selection only sees the event if no gizmo consumed it.
    viewer.input().onMouseButton([&](int button, int action, bool shift, bool ctrl, bool alt) {
        if (button != GLFW_MOUSE_BUTTON_LEFT || ImGui::GetIO().WantCaptureMouse)
            return;

        const bool wasInteracting = gizmoManager.isInteracting();
        gizmoManager.mouseButtonCallback(button, action, shift, ctrl, alt);

        if (wasInteracting || gizmoManager.isInteracting() || !displayPoints)
            return;

        selectionManager.mouseButtonCallback(button, action, shift, ctrl, alt);
    });

    viewer.input().onKeyPress([&](int key, int action, bool shift, bool ctrl, bool alt) {
        if (key != GLFW_KEY_TAB || action != GLFW_PRESS)
            return;
        if (ImGui::GetIO().WantCaptureKeyboard)
            return;

        displayPoints = !displayPoints;
        overlayPointsPass.setEnabled(displayPoints);

        if (!displayPoints)
            selectionManager.clearSelection();
    });

    // -------------------------------------------------------------------------
    // Per-frame callbacks
    // -------------------------------------------------------------------------

    viewer.onGui([&sceneObjects, &lightUploader]() {
        ImGui::Begin("Scene Hierarchy");
        
        int id = 0;
        for (auto &object : sceneObjects)
        {
            ImGui::PushID(id++);
            object->onGUI();
            ImGui::PopID();
        }
        
        ImGui::End();
    });

    lr::SphericalCameraController cameraController(*camera, viewer.input());
    viewer.onUpdate([&cameraController](float dt, VkExtent2D extent) {
        cameraController.update(dt);
    });

    viewer.onUpdate([&](float dt, VkExtent2D extent) {
        gizmoManager.updateCallback(dt, extent, viewer.hasRenderedAtLeastOneFrame(), gizmoReadback, viewer.resources());
    });

    viewer.onUpdate([&](float dt, VkExtent2D extent) {
        selectionManager.updateCallback(dt, extent);
    });

    // Keeps the translate gizmos positioned at the selection centroid, shown
    // only while something is selected, and pushes the result to the overlay pass.
    viewer.onUpdate([&](float dt, VkExtent2D extent) {
        aspect = (extent.height == 0)
            ? 1.0f
            : static_cast<float>(extent.width) / static_cast<float>(extent.height);

        const auto &selected = selectionManager.getSelectedIndices();

        if (selected.empty())
        {
            for (int id : translateGizmoIds)
                gizmoManager.hideGizmo(id);
        }
        else
        {
            glm::vec3 centroid(0.0f);
            for (uint32_t idx : selected)
                centroid += vertexManager.getPositions()[idx];
            centroid /= static_cast<float>(selected.size());

            // Keep the gizmo a constant size on screen (~1/9 screen height) regardless of camera distance.
            const glm::vec3 camPos = camera->getComponent<lr::Transform>().position();
            const float     d      = glm::length(camPos - centroid);
            const float     fov    = glm::radians(camera->getComponent<lr::Camera>().fovYDegrees);
            const float     len    = 2.0f * d * std::tan(fov * 0.5f) / 5.0f;
            const float     rad    = len * 0.45f;

            for (size_t i = 0; i < translateGizmoIds.size(); ++i)
            {
                gizmoManager.unhideGizmo(translateGizmoIds[i]);
                lr::Gizmo &gizmo = gizmoManager.getGizmo(translateGizmoIds[i]);
                gizmo.setPosition(centroid);
                // First 3 gizmos are the X/Y/Z arrows, the 4th is the screen-plane box.
                gizmo.setScale(i < 3 ? glm::vec3(rad, len, rad)
                                     : glm::vec3(rad * 0.45f, rad * 0.45f, rad * 0.45f));
            }
        }

        overlayGeometryPass.setInstances(gizmoManager.getVisibleGizmoInstances());
    });

    viewer.run();
    return 0;
}
catch (const std::exception &e)
{
    spdlog::error("Fatal: {}", e.what());
    throw;
    return 1;
}
