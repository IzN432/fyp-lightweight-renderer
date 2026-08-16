#include "core/app/Viewer.hpp"
#include "core/loaders/GltfLoader.hpp"
#include "core/loaders/Material.hpp"
#include "core/loaders/MaterialStore.hpp"
#include "core/overlay/OverlayMesh.hpp"
#include "core/passes/final/FinalPass.hpp"
#include "core/passes/geometry/GeometryPass.hpp"
#include "core/passes/ibl/IblPass.hpp"
#include "core/passes/pbr/PbrPass.hpp"
#include "core/passes/ambientocclusion/AmbientOcclusionPass.hpp"
#include "core/passes/overlaygeometry/OverlayGeometryPass.hpp"
#include "core/passes/overlaypoints/OverlayPointsPass.hpp"
#include "core/framegraph/ImageReadback.hpp"

#include "core/scene/AreaLightVisual.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/StaticMesh.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/editor/camera/SphericalCameraController.hpp"
#include "core/editor/gizmo/GizmoManager.hpp"
#include "core/editor/gizmo/translate/TranslateArrowGizmo.hpp"
#include "core/editor/gizmo/translate/TranslateBoxGizmo.hpp"
#include "core/editor/selection/BoxSelectionTool.hpp"
#include "core/editor/selection/SelectionManager.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/scene/SceneManager.hpp"
#include "core/scene/Scene.hpp"

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

    {
        lr::IBLPass iblPass({
            .hdriPath  = "C:\\Users\\seani\\Downloads\\cedar_bridge_sunset_2_4k.hdr",
            .envRes    = 2048,
            .irrRes    = 32,
            .pfRes     = 2048,
            .pfMips    = 8
        });
        iblPass.uploadResources(viewer.resources());
        iblPass.preprocess(viewer.frameGraph());
    }

    // -------------------------------------------------------------------------
    // Scene setup
    // -------------------------------------------------------------------------

    lr::Scene scene;

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

    // Flat, up-front reservation for the MaterialStore's GPU-side buffer/texture-array capacity —
    // growing this would mean rebuilding the frame graph's descriptor sets (see MaterialStore.hpp),
    // so it's a generous constant rather than something computed tightly from scene content.
    constexpr uint32_t kMaterialCapacity = 256;

    lr::SceneManager sceneManager(viewer.resources(), kMaterialCapacity, [config]() {
        lr::Material material;
        material.name = "Unused Material Slot";
        material.parameters[config.baseDiffuseName]   = lr::MaterialParam::ColorRGBA{glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)};
        material.parameters[config.baseEmissiveName]  = lr::MaterialParam::ColorRGB{glm::vec3(0.0f)};
        material.parameters[config.baseRoughnessName] = lr::MaterialParam::NormalizedFloat{1.0f};
        material.parameters[config.baseMetallicName]  = lr::MaterialParam::NormalizedFloat{0.0f};
        return material;
    });
    sceneManager.setScene(scene);

    lr::SceneObject* camera = &scene.createSceneObject();
    camera->addComponent<lr::Camera>();
    camera->addComponent<lr::Transform>();
    camera->name = "Main Camera";
    sceneManager.setDefaultCamera(*camera);

    // LIGHT
    {
        lr::DirectionalLight light;
        light.color = glm::vec3(1.0f, 1.0f, 1.0f);
        light.intensity = 1.0f;

        lr::SceneObject& lightObject = scene.createSceneObject();
        lightObject.addComponent<lr::Transform>();
        lightObject.addComponent<lr::Light>(light);
        lightObject.name = "Light";
    }

    // MESH
    const fs::path meshPath = "D:\\FYP\\lion_head_4k.blend\\lion_head_4k.glb";

    lr::GltfLoader gltfLoader;
    auto [sequence, materialHandles] = gltfLoader.load(meshPath, sceneManager.materialStore(), config);

    if (sequence.empty())
        throw std::runtime_error("GltfLoader returned empty sequence for '" + meshPath.string() + "'");

    // LIGHT VISUALS — every light, not just ones that start out as AreaLight, gets its own StaticMesh
    // component (a quad), separate from the main mesh's StaticMesh. The quad still draws through the
    // same GeometryPass as the main mesh (see AreaLightVisual.hpp for why the visual needs to be real
    // geometry rather than an overlay); its material lives in a MaterialStore slot acquired up front,
    // so switching a light's type at runtime (see Light::onGUIImpl) just rewrites that slot in place —
    // see SceneManager::updateLightVisuals.
    const lr::AreaLightVisualConfig areaLightVisualConfig{
        .normalAttributeName  = config.normalAttributeName,
        .tangentAttributeName = config.tangentAttributeName,
        .uvAttributeName      = config.uvAttributeName,
        .baseDiffuseName      = config.baseDiffuseName,
        .baseEmissiveName     = config.baseEmissiveName,
        .baseRoughnessName    = config.baseRoughnessName,
        .baseMetallicName     = config.baseMetallicName,
    };

    lr::SceneObject* meshObject = &scene.createSceneObject();
    meshObject->addComponent<lr::Transform>();
    {
        // Seeds the main mesh's selection-highlight colors — one per unique/deduped position, the
        // same space VertexManager/SelectionManager and the points-picking overlay operate in.
        // SceneManager::uploadMeshes() reads this back to build the initial GPU color buffer, so it
        // must be set before sceneManager.initialize() runs.
        lr::Mesh &m = sequence.frames.front();
        std::vector<glm::vec3> colors(m.positions.size(), glm::vec3(1.0f, 0.0f, 1.0f));
        m.setPerUniqueVertexArray("color", std::span<const glm::vec3>(colors));
    }
    auto &staticMesh = meshObject->addComponent<lr::StaticMesh>(sequence.frames.front(), materialHandles,
                                                                 sceneManager.materialStore());
    meshObject->name = "Mesh Object";
    sceneManager.setMainMeshObject(*meshObject);

    // -------------------------------------------------------------------------
    // Resource uploads
    // -------------------------------------------------------------------------

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

    // Builds light visuals, uploads the initial lights/mesh/material/camera buffers, and wires the
    // change listeners that keep the camera UBO and main mesh's materials SSBO in sync afterward —
    // see SceneManager::initialize().
    sceneManager.initialize(areaLightVisualConfig, gpuMaterialLayout,
                            { config.normalAttributeName, config.tangentAttributeName, config.uvAttributeName },
                            viewer.input());

    // -------------------------------------------------------------------------
    // Frame graph passes
    // -------------------------------------------------------------------------

    const VkFormat swapchainFormat =
        viewer.frameGraph().resources().getImage("swapchain")->format;

    lr::GeometryPass geometryPass({
        .cameraBufferResourceName  = sceneManager.cameraBufferName(),
        .vertexBufferResourceNames = { {0, sceneManager.mainMeshPositionBufferName()}, {1, sceneManager.mainMeshVertexBufferName()} },
        .vertexBufferUploadResult  = sceneManager.meshPositions(),
        .indexBufferUploadResult   = sceneManager.indexBuffer(),
        .meshTransforms = sceneManager.meshTransforms(),
        .indexBufferResourceName = sceneManager.mainMeshIndexBufferName(),
        .faceGroupBufferResourceName = sceneManager.mainMeshFaceGroupBufferName(),
        .diffuseTextureArrayResourceName = sceneManager.materialUploadResult().textureNameMap.at(config.diffuseTextureName),
        .normalTextureArrayResourceName = sceneManager.materialUploadResult().textureNameMap.at(config.normalTextureName),
        .metallicRoughnessTextureArrayResourceName = sceneManager.materialUploadResult().textureNameMap.at(config.metallicRoughnessTextureName),
        .emissiveTextureArrayResourceName = sceneManager.materialUploadResult().textureNameMap.at(config.emissiveTextureName),
        .materialBufferResourceName = sceneManager.materialUploadResult().materialInfoBufferName,

        .materialCount = sceneManager.materialStore().capacity(),
    });
    lr::GpuMeshLayout gpuMeshLayout(staticMesh.mesh().layout());

    gpuMeshLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT);
    gpuMeshLayout.map(config.normalAttributeName,  1, 1, VK_FORMAT_R32G32B32_SFLOAT);
    gpuMeshLayout.map(config.tangentAttributeName, 1, 2, VK_FORMAT_R32G32B32A32_SFLOAT);
    gpuMeshLayout.map(config.uvAttributeName,      1, 3, VK_FORMAT_R32G32_SFLOAT);

    geometryPass.build(viewer.frameGraph(), gpuMeshLayout);

    lr::AmbientOcclusionPass aoPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
    });
    aoPass.uploadResources(viewer.resources());
    aoPass.build(viewer.frameGraph());

    lr::PbrPass pbrPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
        .lightBufferResourceName = sceneManager.lightBufferName(),
        .numLights = sceneManager.numLights(),
        .pfMips = 8,
    });
    pbrPass.uploadResources(viewer.resources());
    pbrPass.build(viewer.frameGraph());

    lr::OverlayGeometryPass overlayGeometryPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
    });
    overlayGeometryPass.uploadResources(viewer.resources());
    overlayGeometryPass.build(viewer.frameGraph());
    overlayGeometryPass.setInstances({});

    lr::GpuMeshLayout pointsMeshLayout(staticMesh.mesh().layout());
    pointsMeshLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT);
    pointsMeshLayout.mapUniqueVertex("color", 0, 1, VK_FORMAT_R32G32B32_SFLOAT);

    lr::OverlayPointsPass overlayPointsPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
        .pointsBufferResourceName = sceneManager.mainMeshPointsBufferName(),
        .pointsBufferUploadResult = sceneManager.mainMeshPoints(),
        .vertexCounts             = { static_cast<uint32_t>(staticMesh.mesh().positions.size()) },
        .meshTransform            = &meshObject->getComponent<lr::Transform>(),
    });
    overlayPointsPass.build(viewer.frameGraph(), pointsMeshLayout);

    lr::FinalPass finalPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
        .swapchainFormat = swapchainFormat,
    });
    finalPass.build(viewer.frameGraph());

    // -------------------------------------------------------------------------
    // Editor state — vertex picking, selection and gizmo managers
    // -------------------------------------------------------------------------

    overlayPointsPass.setEnabled(sceneManager.selectionState() == lr::SelectionState::Edit);

    // Gizmo hover — reads the picking image from the previous frame
    lr::ImageReadback gizmoReadback(viewer.context(), viewer.allocator());

    lr::VertexManager vertexManager(staticMesh.mesh().positions);
    vertexManager.registerUpdateCallback([&]() {
        sceneManager.updateMainMeshPositions();
    });

    lr::CommandManager commandManager;

    // SceneManager owns the SelectionManager (constructed off the main mesh in initialize(), see
    // SceneManager::selectionManager()) since it needs to wire selection-highlight changes straight
    // to the GPU color buffer; this is just a local alias to keep the call sites below unchanged.
    lr::SelectionManager &selectionManager = sceneManager.selectionManager();
    selectionManager.setSelectTool(std::make_unique<lr::BoxSelectionTool>(viewer.input(), *camera, selectionManager));

    lr::GizmoManager gizmoManager(overlayGeometryPass, viewer.input());
    const std::vector<int> translateGizmoIds = {
        gizmoManager.addGizmo(std::make_unique<lr::TranslateArrowGizmo>(
            lr::TranslateArrowGizmoAxis::X, *camera, viewer.input(), vertexManager, selectionManager, commandManager)),
        gizmoManager.addGizmo(std::make_unique<lr::TranslateArrowGizmo>(
            lr::TranslateArrowGizmoAxis::Y, *camera, viewer.input(), vertexManager, selectionManager, commandManager)),
        gizmoManager.addGizmo(std::make_unique<lr::TranslateArrowGizmo>(
            lr::TranslateArrowGizmoAxis::Z, *camera, viewer.input(), vertexManager, selectionManager, commandManager)),
        gizmoManager.addGizmo(std::make_unique<lr::TranslateBoxGizmo>(
            *camera, viewer.input(), vertexManager, selectionManager, commandManager)),
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

        if (wasInteracting || gizmoManager.isInteracting() || sceneManager.selectionState() != lr::SelectionState::Edit)
            return;

        selectionManager.mouseButtonCallback(button, action, shift, ctrl, alt);
    });

    viewer.input().onKeyPress([&](int key, int action, bool shift, bool ctrl, bool alt) {
        if (key != GLFW_KEY_TAB || action != GLFW_PRESS)
            return;
        if (ImGui::GetIO().WantCaptureKeyboard)
            return;

        const bool nowEditing = sceneManager.selectionState() != lr::SelectionState::Edit;
        sceneManager.setSelectionState(nowEditing ? lr::SelectionState::Edit : lr::SelectionState::View);
        overlayPointsPass.setEnabled(nowEditing);
    });

    viewer.input().onKeyPress([&](int key, int action, bool shift, bool ctrl, bool alt) {
        if (key != GLFW_KEY_Z || action != GLFW_PRESS || !ctrl)
            return;
        if (ImGui::GetIO().WantCaptureKeyboard)
            return;

        commandManager.undo();
    });
    // -------------------------------------------------------------------------
    // Per-frame callbacks
    // -------------------------------------------------------------------------

    viewer.onGui([&scene]() {
        ImGui::Begin("Scene Hierarchy");

        int id = 0;
        for (auto &object : scene.sceneObjects())
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

    // selectionManager's per-frame mouse/drag handling is driven by SceneManager::registerCallbacks()
    // below, since SceneManager now owns the SelectionManager instance.

    // Keeps the translate gizmos positioned at the selection centroid, shown
    // only while something is selected, and pushes the result to the overlay pass.
    viewer.onUpdate([&](float dt, VkExtent2D extent) {
        const auto &selected = selectionManager.getSelectedIndices();

        if (selected.empty())
        {
            for (int id : translateGizmoIds)
                gizmoManager.hideGizmo(id);
        }
        else
        {
            // Average in local space, then transform once — valid since centroid-of-transformed-points
            // equals transform-of-centroid for any affine map. The gizmo itself is positioned in world
            // space (it's not part of the mesh, so GeometryPass's model matrix never applies to it),
            // so it needs to track where the selected vertices actually render, not their local positions.
            glm::vec3 localCentroid(0.0f);
            for (uint32_t idx : selected)
                localCentroid += vertexManager.getPositions()[idx];
            localCentroid /= static_cast<float>(selected.size());
            const glm::vec3 centroid = glm::vec3(
                meshObject->getComponent<lr::Transform>().localMatrix() * glm::vec4(localCentroid, 1.0f));

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

    // Registers SceneManager's own onUpdate (aspect tracking) and onLateUpdate (flushDirty —
    // runs after every onUpdate above, so it sees the results of this frame's camera
    // controller / gizmo / GUI edits and does at most one GPU re-upload per dirtied resource
    // rather than one per individual mutation) callbacks — see SceneManager::registerCallbacks.
    sceneManager.registerCallbacks(viewer);

    viewer.run();
    return 0;
}
catch (const std::exception &e)
{
    spdlog::error("Fatal: {}", e.what());
    throw;
    return 1;
}
