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

#include "core/scene/Camera.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/StaticMesh.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/upload/CameraUploader.hpp"
#include "core/upload/LightUploader.hpp"
#include "core/upload/MaterialUploader.hpp"
#include "core/upload/MeshUploader.hpp"
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

    // CAMERA — spherical orbit state (Blender-style)
    glm::vec3 orbitTarget(0.0f);
    float orbitRadius    = 5.0f;
    float orbitAzimuth   = 0.0f;   // radians; 0 = camera on +Z axis
    float orbitElevation = 0.0f;   // radians; 0 = horizontal

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
    // Point picking state
    // -------------------------------------------------------------------------

    std::vector<uint32_t> selectedVertices;
    bool selectionDirty = false;

    // Pending pick state set by the mouse callback, consumed in onUpdate.
    bool     pickPending    = false;
    bool     boxPickPending = false;
    uint32_t pickX          = 0;
    uint32_t pickY          = 0;
    uint32_t boxPickX0      = 0;
    uint32_t boxPickY0      = 0;
    uint32_t boxPickX1      = 0;
    uint32_t boxPickY1      = 0;
    bool     pickShift      = false;
    double   mouseDownX     = 0.0;
    double   mouseDownY     = 0.0;

    static constexpr double kPickDragThreshold = 4.0;

    viewer.input().onMouseButton([&](int button, int action) {
        if (button != GLFW_MOUSE_BUTTON_LEFT)
            return;

        double mx, my;
        viewer.input().getMousePos(mx, my);

        if (action == GLFW_PRESS)
        {
            mouseDownX = mx;
            mouseDownY = my;
        }
        else if (action == GLFW_RELEASE)
        {
            double ddx = mx - mouseDownX;
            double ddy = my - mouseDownY;
            pickShift = viewer.input().isKeyPressed(GLFW_KEY_LEFT_SHIFT) ||
                        viewer.input().isKeyPressed(GLFW_KEY_RIGHT_SHIFT);

            if (std::abs(ddx) < kPickDragThreshold && std::abs(ddy) < kPickDragThreshold)
            {
                pickPending = true;
                pickX       = static_cast<uint32_t>(mouseDownX);
                pickY       = static_cast<uint32_t>(mouseDownY);
            }
            else
            {
                boxPickPending = true;
                boxPickX0 = static_cast<uint32_t>(std::min(mouseDownX, mx));
                boxPickY0 = static_cast<uint32_t>(std::min(mouseDownY, my));
                boxPickX1 = static_cast<uint32_t>(std::max(mouseDownX, mx));
                boxPickY1 = static_cast<uint32_t>(std::max(mouseDownY, my));
            }
        }
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

    viewer.onUpdate([&](float dt, VkExtent2D extent) {
        aspect = (extent.height == 0)
            ? 1.0f
            : static_cast<float>(extent.width) / static_cast<float>(extent.height);

        // ---- CPU-side vertex picking ----
        // Project all vertices to screen space and either find the nearest one
        // (single click) or collect all that fall inside the drag rect (box).
        // Both paths run fully on CPU — no GPU readback stall.
        if ((pickPending || boxPickPending) && !ImGui::GetIO().WantCaptureMouse)
        {
            const glm::mat4 vp = camera->getComponent<lr::Camera>().viewProjectionMatrix(aspect);
            const float     sw = static_cast<float>(extent.width);
            const float     sh = static_cast<float>(extent.height);

            if (pickPending)
            {
                pickPending = false;

                static constexpr float kPickRadius = 12.0f;

                const float cx = static_cast<float>(pickX);
                const float cy = static_cast<float>(pickY);

                uint32_t bestIdx  = ~0u;
                float    bestDist = kPickRadius * kPickRadius;

                for (uint32_t i = 0; i < staticMesh.mesh().positions.size(); ++i)
                {
                    const glm::vec4 clip = vp * glm::vec4(staticMesh.mesh().positions[i], 1.0f);
                    if (clip.w <= 0.0f) continue;
                    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
                    if (ndc.z < 0.0f || ndc.z > 1.0f) continue;

                    const float sx = (ndc.x * 0.5f + 0.5f) * sw;
                    const float sy = (ndc.y * 0.5f + 0.5f) * sh;
                    const float d2 = (sx - cx) * (sx - cx) + (sy - cy) * (sy - cy);
                    if (d2 < bestDist) { bestDist = d2; bestIdx = i; }
                }

                if (!pickShift)
                    selectedVertices.clear();

                if (bestIdx != ~0u)
                {
                    auto it = std::find(selectedVertices.begin(), selectedVertices.end(), bestIdx);
                    if (it != selectedVertices.end())
                        selectedVertices.erase(it);  // shift-click deselects
                    else
                        selectedVertices.push_back(bestIdx);
                }

                selectionDirty = true;
            }

            if (boxPickPending)
            {
                boxPickPending = false;

                const float bx0 = static_cast<float>(boxPickX0);
                const float by0 = static_cast<float>(boxPickY0);
                const float bx1 = static_cast<float>(boxPickX1);
                const float by1 = static_cast<float>(boxPickY1);

                if (!pickShift)
                    selectedVertices.clear();

                for (uint32_t i = 0; i < staticMesh.mesh().positions.size(); ++i)
                {
                    const glm::vec4 clip = vp * glm::vec4(staticMesh.mesh().positions[i], 1.0f);
                    if (clip.w <= 0.0f) continue;
                    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
                    if (ndc.z < 0.0f || ndc.z > 1.0f) continue;

                    const float sx = (ndc.x * 0.5f + 0.5f) * sw;
                    const float sy = (ndc.y * 0.5f + 0.5f) * sh;

                    if (sx >= bx0 && sx <= bx1 && sy >= by0 && sy <= by1)
                    {
                        if (std::find(selectedVertices.begin(), selectedVertices.end(), i) ==
                            selectedVertices.end())
                            selectedVertices.push_back(i);
                    }
                }

                selectionDirty = true;
            }
        }

        // Update color buffer and XYZ gizmo whenever the selection changes
        if (selectionDirty)
        {
            selectionDirty = false;

            std::fill(pointColors.begin(), pointColors.end(), glm::vec3(1.0f, 0.0f, 1.0f));
            for (uint32_t idx : selectedVertices)
                pointColors[idx] = glm::vec3(1.0f, 0.8f, 0.0f);  // orange = selected
            viewer.resources().updateBuffer(
                mainMeshColorBufferName,
                pointColors.data(),
                pointColors.size() * sizeof(glm::vec3));

            // Spawn an XYZ arrow gizmo at the centroid of the selected points
            std::vector<lr::OverlayInstance> gizmoInstances;
            if (!selectedVertices.empty())
            {
                glm::vec3 centroid(0.0f);
                for (uint32_t idx : selectedVertices)
                    centroid += staticMesh.mesh().positions[idx];
                centroid /= static_cast<float>(selectedVertices.size());

                constexpr float len = 0.2f;
                constexpr float rad = 0.025f;
                constexpr float occ = 0.3f;

                // Arrow points +Y by default; rotate to each world axis
                gizmoInstances = {
                    {   // +X  (red):   euler Z=-90 rotates +Y to +X
                        .primitive       = lr::OverlayPrimitive::Arrow,
                        .position        = centroid,
                        .eulerDegrees    = {0.0f, 0.0f, -90.0f},
                        .scale           = {rad, len, rad},
                        .color           = {1.0f, 0.0f, 0.0f},
                        .occludedOpacity = occ,
                    },
                    {   // +Y  (green): no rotation needed
                        .primitive       = lr::OverlayPrimitive::Arrow,
                        .position        = centroid,
                        .eulerDegrees    = {0.0f, 0.0f, 0.0f},
                        .scale           = {rad, len, rad},
                        .color           = {0.0f, 1.0f, 0.0f},
                        .occludedOpacity = occ,
                    },
                    {   // +Z  (blue):  euler X=+90 rotates +Y to +Z
                        .primitive       = lr::OverlayPrimitive::Arrow,
                        .position        = centroid,
                        .eulerDegrees    = {90.0f, 0.0f, 0.0f},
                        .scale           = {rad, len, rad},
                        .color           = {0.0f, 0.0f, 1.0f},
                        .occludedOpacity = occ,
                    },
                };
            }
            overlayGeometryPass.setInstances(std::move(gizmoInstances));
        }

        double dx, dy;
        viewer.input().getMouseDelta(dx, dy);
        double scroll = viewer.input().getScrollDelta();
        bool mmb   = viewer.input().isMouseButtonPressed(GLFW_MOUSE_BUTTON_MIDDLE);
        bool shift = viewer.input().isKeyPressed(GLFW_KEY_LEFT_SHIFT) ||
                     viewer.input().isKeyPressed(GLFW_KEY_RIGHT_SHIFT);

        bool reset = viewer.input().isKeyPressed(GLFW_KEY_R);
        if (reset) {
            orbitTarget = glm::vec3(0.0f);
            orbitRadius = 5.0f;
            orbitAzimuth = 0.0f;
            orbitElevation = 0.0f;
        }
        
        if (!ImGui::GetIO().WantCaptureMouse) {
            if (mmb && shift) {
                // Pan: translate target in camera right/up plane
                auto &t = camera->getComponent<lr::Transform>();
                float panSpeed = orbitRadius * 0.002f;
                orbitTarget -= t.right() * (float)dx * panSpeed;
                orbitTarget += t.up()    * (float)dy * panSpeed;
            } else if (mmb) {
                orbitAzimuth   -= (float)dx * 0.01f;
                orbitElevation -= (float)dy * 0.01f;
                orbitElevation  = glm::clamp(orbitElevation,
                                    glm::radians(-89.0f), glm::radians(89.0f));
            }

            if (scroll != 0.0) {
                orbitRadius *= std::pow(1.0f / 1.1f, (float)scroll);
                orbitRadius  = glm::clamp(orbitRadius, 0.01f, 1000.0f);
            }
        } // !WantCaptureMouse

        glm::vec3 pos(
            orbitTarget.x + orbitRadius * std::cos(orbitElevation) * std::sin(orbitAzimuth),
            orbitTarget.y + orbitRadius * std::sin(orbitElevation),
            orbitTarget.z + orbitRadius * std::cos(orbitElevation) * std::cos(orbitAzimuth));

        camera->getComponent<lr::Transform>().position = pos;
        const glm::mat4 view = glm::lookAt(pos, orbitTarget, glm::vec3(0.0f, 1.0f, 0.0f));
        camera->getComponent<lr::Transform>().setRotation(glm::conjugate(glm::quat_cast(view)));

        updateCameraUpload();
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
