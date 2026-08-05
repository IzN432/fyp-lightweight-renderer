#include "core/app/Viewer.hpp"
#include "core/loaders/GltfLoader.hpp"
#include "core/loaders/ObjLoader.hpp"
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
#include <ImGuiFileDialog.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec4.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
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
    {
        lr::DirectionalLight light;
        light.color = glm::vec3(1.0f, 1.0f, 1.0f);
        light.intensity = 1.0f;
        
        lr::SceneObject* lightObject = sceneObjects.emplace_back(std::make_unique<lr::SceneObject>()).get();
        lightObject->addComponent<lr::Transform>();
        lightObject->addComponent<lr::Light>(light);
        lightObject->name = "Directional Light";
    }

    // AREA LIGHT — rendered both as an LTC light (see pbr.frag CalcAreaLight) and, further down,
    // as a real emissive quad mesh so it's visible when looked at directly (see AreaLightVisual.hpp).
    {
        lr::AreaLight areaLight;
        areaLight.color = glm::vec3(1.0f, 0.6f, 0.3f);
        areaLight.intensity = 4.0f;
        areaLight.size = glm::vec2(1.5f, 1.0f);

        lr::SceneObject* areaLightObject = sceneObjects.emplace_back(std::make_unique<lr::SceneObject>()).get();
        areaLightObject->addComponent<lr::Transform>(
            glm::vec3(0.0f, 2.0f, 2.0f),
            glm::quat(glm::radians(glm::vec3(-45.0f, 180.0f, 0.0f))));
        areaLightObject->addComponent<lr::Light>(areaLight);
        areaLightObject->name = "Area Light";
    }

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

    // OBJ files (see addMeshFromFile below) use their own loader, but the output key names are
    // set to match `config` above so both loaders' materials share the one GpuMaterialLayout
    // built later (gpuMaterialLayout). ambient/specular/shininess have no corresponding scalar/
    // texture in that layout and are simply unused.
    lr::ObjLoader objLoader;
    lr::ObjLoaderConfig objConfig{
        .normalAttributeName = config.normalAttributeName,
        .tangentAttributeName = config.tangentAttributeName,
        .uvAttributeName = config.uvAttributeName,
        .diffuseTextureName = config.diffuseTextureName,
        .normalTextureName = config.normalTextureName,
        .metallicRoughnessTextureName = config.metallicRoughnessTextureName,
        .emissiveTextureName = config.emissiveTextureName,
        .baseDiffuseName = config.baseDiffuseName,
        .baseRoughnessName = config.baseRoughnessName,
        .baseMetallicName = config.baseMetallicName,
        .baseEmissiveName = config.baseEmissiveName,
    };

    // LIGHT VISUALS — every light, not just ones that start out as AreaLight, gets its own StaticMesh
    // component (a quad), separate from the main mesh's StaticMesh. The quad still draws through the
    // same GeometryPass as the main mesh (see AreaLightVisual.hpp for why the visual needs to be real
    // geometry rather than an overlay), which means its material index has to be baked in as a global
    // index into the combined material buffer built below — see lightVisualMaterialBase.
    //
    // Pre-allocating a slot for every light (rather than only ones currently typed AreaLight) is what
    // makes this robust to a light's type being changed at runtime via the Scene Hierarchy's type
    // combo (see Light::onGUIImpl): the shared geometry/material buffers never need to grow or shrink
    // when that happens — updateLightVisuals below just rewrites what's in an already-existing slot.
    std::vector<lr::SceneObject*> lightVisualObjects;
    for (const auto &object : sceneObjects)
        if (object->hasComponent<lr::Light>())
            lightVisualObjects.push_back(object.get());

    const lr::AreaLightVisualConfig areaLightVisualConfig{
        .normalAttributeName  = config.normalAttributeName,
        .tangentAttributeName = config.tangentAttributeName,
        .uvAttributeName      = config.uvAttributeName,
        .baseDiffuseName      = config.baseDiffuseName,
        .baseEmissiveName     = config.baseEmissiveName,
        .baseRoughnessName    = config.baseRoughnessName,
        .baseMetallicName     = config.baseMetallicName,
    };

    // A light whose current type isn't AreaLight draws as a degenerate, zero-emissive quad rather
    // than std::get-ing a variant that isn't AreaLight.
    static const lr::AreaLight hiddenAreaLightVisual{ {glm::vec3(0.0f), 0.0f}, glm::vec2(0.0f) };

    // Base index, in the combined material buffer, where the light visuals' materials start — the
    // glTF materials (indices [0, lightVisualMaterialBase)) come first, then one material per light.
    const uint32_t lightVisualMaterialBase = static_cast<uint32_t>(materials.size());
    for (size_t i = 0; i < lightVisualObjects.size(); ++i)
    {
        const auto *areaLight = std::get_if<lr::AreaLight>(&lightVisualObjects[i]->getComponent<lr::Light>().light);
        const auto &lightData = areaLight ? *areaLight : hiddenAreaLightVisual;
        const auto &transform = lightVisualObjects[i]->getComponent<lr::Transform>();

        lr::Mesh quadMesh;
        lr::buildAreaLightQuadMesh(quadMesh, transform, lightData,
                                    lightVisualMaterialBase + static_cast<uint32_t>(i), areaLightVisualConfig);
        std::vector<lr::Material> quadMaterials;
        quadMaterials.push_back(lr::buildAreaLightMaterial(lightData, areaLightVisualConfig));

        lightVisualObjects[i]->addComponent<lr::StaticMesh>(quadMesh, quadMaterials, /*hideFromGui=*/true);
    }

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

    // Change-listener wiring for these lights (updateLightList, plus updateMaterialUpload and
    // updateLightVisuals below) happens later, via wireLightObject, once all three listener
    // targets exist — see the wireLightObject definition after updateLightVisuals.
    std::vector<lr::SceneObject*> sceneLights;
    for (const auto &sceneObject : sceneObjects)
        if (sceneObject->hasComponent<lr::Light>())
            sceneLights.push_back(sceneObject.get());

    lightUploader.upload(sceneLights);
    
    // Upload the main scene geometry — the area light quad meshes are drawn through the same
    // GeometryPass, so they share these buffers with the main mesh (see GeometryPass.cpp, which
    // loops over one drawIndexed per mesh out of a single shared vertex/index buffer set).
    lr::MeshUploader meshUploader(viewer.resources());
    const std::string mainMeshPositionBufferName  = "meshPositionBuffer";
    const std::string mainMeshVertexBufferName    = "meshVertexBuffer";
    const std::string mainMeshColorBufferName     = "meshColorBuffer";
    const std::string mainMeshIndexBufferName     = "meshIndexBuffer";
    const std::string mainMeshFaceGroupBufferName = "meshFaceGroupBuffer";

    // Objects loaded into the scene at runtime (see addMeshFromFile below) — kept separate from
    // lightVisualObjects since they're plain renderable meshes, not light-driven. localFaceGroups
    // is a pristine, pre-offset copy of the loaded mesh's face groups: buildCombinedMaterials()
    // below places lightVisualObjects' materials before extraModelObjects', so growing
    // lightVisualObjects at runtime (adding a light) shifts the absolute material offset every
    // extraModelObject was baked against — rebakeExtraModelFaceGroups() (defined after
    // rebuildSceneGeometry) uses this local copy to recompute that offset from scratch.
    struct ExtraModelEntry
    {
        lr::SceneObject *object;
        std::vector<uint32_t> localFaceGroups;
    };
    std::vector<ExtraModelEntry> extraModelObjects;

    // Gathers every mesh that shares the buffers below (main mesh + light visuals + anything
    // loaded at runtime), in the fixed order the combined material indices assume. Called once to
    // build the initial geometryMeshes list, and again from rebuildSceneGeometry() whenever that
    // set changes.
    auto currentGeometryMeshes = [&]() {
        std::vector<const lr::Mesh*> meshes = { &staticMesh.mesh() };
        for (auto *lightVisualObject : lightVisualObjects)
            meshes.push_back(&lightVisualObject->getComponent<lr::StaticMesh>().mesh());
        for (auto &extraModelEntry : extraModelObjects)
            meshes.push_back(&extraModelEntry.object->getComponent<lr::StaticMesh>().mesh());
        return meshes;
    };

    // Parallel to currentGeometryMeshes() (same order, one entry per mesh) — the model matrix
    // GeometryPass applies per draw. Light visuals bake their Transform into vertex positions
    // directly (see AreaLightVisual.hpp), so they'd be double-transformed by also applying their
    // Transform here — pass nullptr for those to mean "draw with an identity model matrix".
    auto currentMeshTransforms = [&]() {
        std::vector<const lr::Transform*> transforms = { &meshObject->getComponent<lr::Transform>() };
        for (size_t i = 0; i < lightVisualObjects.size(); ++i)
            transforms.push_back(nullptr);
        for (auto &extraModelEntry : extraModelObjects)
            transforms.push_back(&extraModelEntry.object->getComponent<lr::Transform>());
        return transforms;
    };

    std::vector<const lr::Mesh*> geometryMeshes = currentGeometryMeshes();

    const lr::VertexBufferUploadConfig meshPositionUploadConfig{
        .vertexBufferName = mainMeshPositionBufferName,
        .includePosition  = true
    };
    const lr::VertexBufferUploadConfig meshAttributeUploadConfig{
        .vertexBufferName     = mainMeshVertexBufferName,
        .vertexAttributeNames = { config.normalAttributeName, config.tangentAttributeName, config.uvAttributeName }
    };

    lr::VertexBufferUploadResult meshPositions = meshUploader.uploadVertexBuffer(geometryMeshes, meshPositionUploadConfig);
    meshUploader.uploadVertexBuffer(geometryMeshes, meshAttributeUploadConfig);
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
    lr::IndexBufferUploadResult indexBuffer = meshUploader.uploadIndexBuffer(
        geometryMeshes,
        { .indexBufferName = mainMeshIndexBufferName });
    meshUploader.uploadFaceGroupBuffer(
        geometryMeshes,
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

    // The GPU material buffer is one flat array, but ownership of the materials themselves is now
    // split across StaticMesh components (the main mesh's, plus one per light, plus one per
    // runtime-loaded model) — this gathers them back into the layout the buffer expects: glTF
    // materials first, then one per light (matching lightVisualMaterialBase above), then one per
    // runtime-loaded model appended at the end (see the "Load Another Model" button below).
    auto buildCombinedMaterials = [&]() {
        std::vector<const lr::Material*> combined;
        for (const auto &m : staticMesh.materials())
            combined.push_back(&m);
        for (auto *lightVisualObject : lightVisualObjects)
            for (const auto &m : lightVisualObject->getComponent<lr::StaticMesh>().materials())
                combined.push_back(&m);
        for (auto &extraModelEntry : extraModelObjects)
            for (const auto &m : extraModelEntry.object->getComponent<lr::StaticMesh>().materials())
                combined.push_back(&m);
        return combined;
    };

    lr::MaterialUploader materialUploader(viewer.resources());
    lr::MaterialUploadResult material = materialUploader.upload(
        buildCombinedMaterials(),
        gpuMaterialLayout,
        "material");

    std::function<void()> updateMaterialUpload = [&]() {
        materialUploader.update(buildCombinedMaterials(), gpuMaterialLayout, material);
    };
    staticMesh.addChangeListener(updateMaterialUpload);

    // Keep each light's visual quad (position/orientation/size) and emissive material in sync
    // whenever that light's Transform or Light component is edited in the Scene Hierarchy — including
    // being switched to a different light type at runtime (see Light::onGUIImpl's type combo), at
    // which point the quad collapses to (or springs from) the hidden zero-sized state.
    std::function<void()> updateLightVisuals = [&]() {
        for (size_t i = 0; i < lightVisualObjects.size(); ++i)
        {
            const auto *areaLight = std::get_if<lr::AreaLight>(&lightVisualObjects[i]->getComponent<lr::Light>().light);
            const auto &lightData = areaLight ? *areaLight : hiddenAreaLightVisual;
            const auto &transform = lightVisualObjects[i]->getComponent<lr::Transform>();
            auto &lightStaticMesh = lightVisualObjects[i]->getComponent<lr::StaticMesh>();

            lr::buildAreaLightQuadMesh(lightStaticMesh.mesh(), transform, lightData,
                                        lightVisualMaterialBase + static_cast<uint32_t>(i), areaLightVisualConfig);
            lightStaticMesh.materials()[0] = lr::buildAreaLightMaterial(lightData, areaLightVisualConfig);
        }

        meshUploader.updateVertexBuffer(geometryMeshes, meshPositionUploadConfig);
        meshUploader.updateVertexBuffer(geometryMeshes, meshAttributeUploadConfig);
        updateMaterialUpload();
    };

    // Wires a light object's GPU-side plumbing: light-buffer re-upload, visual-quad rebuild, and
    // material re-upload for that quad. Used both for the two startup lights below and for any
    // light added later via addPointLightObject.
    auto wireLightObject = [&](lr::SceneObject *lightObject) {
        lightObject->getComponent<lr::Light>().addChangeListener(updateLightList);
        lightObject->getComponent<lr::Transform>().addChangeListener(updateLightList);
        lightObject->getComponent<lr::Light>().addChangeListener(updateLightVisuals);
        lightObject->getComponent<lr::Transform>().addChangeListener(updateLightVisuals);
        lightObject->getComponent<lr::StaticMesh>().addChangeListener(updateMaterialUpload);
    };

    for (auto *lightVisualObject : lightVisualObjects)
        wireLightObject(lightVisualObject);

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
        .meshTransforms = currentMeshTransforms(),
        .indexBufferResourceName = mainMeshIndexBufferName,
        .faceGroupBufferResourceName = mainMeshFaceGroupBufferName,
        .diffuseTextureArrayResourceName = material.textureNameMap.at(config.diffuseTextureName),
        .normalTextureArrayResourceName = material.textureNameMap.at(config.normalTextureName),
        .metallicRoughnessTextureArrayResourceName = material.textureNameMap.at(config.metallicRoughnessTextureName),
        .emissiveTextureArrayResourceName = material.textureNameMap.at(config.emissiveTextureName),
        .materialBufferResourceName = material.materialInfoBufferName,

        .materialCount = lightVisualMaterialBase + static_cast<uint32_t>(lightVisualObjects.size()),
    });
    lr::GpuMeshLayout gpuMeshLayout(staticMesh.mesh().layout());

    gpuMeshLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT);
    gpuMeshLayout.map(config.normalAttributeName,  1, 1, VK_FORMAT_R32G32B32_SFLOAT);
    gpuMeshLayout.map(config.tangentAttributeName, 1, 2, VK_FORMAT_R32G32B32A32_SFLOAT);
    gpuMeshLayout.map(config.uvAttributeName,      1, 3, VK_FORMAT_R32G32_SFLOAT);

    geometryPass.build(viewer.frameGraph(), gpuMeshLayout);

    // Rebuilds the shared vertex/index/facegroup/material buffers and texture arrays from scratch
    // to reflect the current set of renderable objects (main mesh + light visuals +
    // extraModelObjects), then re-declares the geometry pass and recompiles the frame graph so its
    // descriptor sets — materialCount is baked into their size — pick up the new content. Used
    // whenever a new object is added to the scene at runtime (see addPointLightObject and
    // addMeshFromFile below). The GPU must be idle before any of this, same as
    // Viewer::recreateSwapchain() (the only other place that tears down GPU resources the render
    // loop might still be using).
    std::function<void()> rebuildSceneGeometry = [&]() {
        geometryMeshes = currentGeometryMeshes();
        const auto combinedMaterials = buildCombinedMaterials();

        viewer.context().waitIdle();

        viewer.resources().destroyBuffer(mainMeshPositionBufferName);
        viewer.resources().destroyBuffer(mainMeshVertexBufferName);
        viewer.resources().destroyBuffer(mainMeshIndexBufferName);
        viewer.resources().destroyBuffer(mainMeshFaceGroupBufferName);
        viewer.resources().destroyBuffer(material.materialInfoBufferName);
        for (const auto &[textureName, resourceName] : material.textureNameMap)
            viewer.resources().destroyImageArray(resourceName);

        meshPositions = meshUploader.uploadVertexBuffer(geometryMeshes, meshPositionUploadConfig);
        meshUploader.uploadVertexBuffer(geometryMeshes, meshAttributeUploadConfig);
        indexBuffer = meshUploader.uploadIndexBuffer(
            geometryMeshes,
            { .indexBufferName = mainMeshIndexBufferName });
        meshUploader.uploadFaceGroupBuffer(
            geometryMeshes,
            { .faceGroupBufferName = mainMeshFaceGroupBufferName });

        material = materialUploader.upload(combinedMaterials, gpuMaterialLayout, "material");

        // FrameGraph::buildBarriers() only emits a transition barrier when the registry's tracked
        // layout for a resource differs from what the next pass needs. The registry has no way to
        // know the swapchain image was handed back to PRESENT_SRC_KHR by Renderer::transitionForPresent()
        // after the last frame (that transition happens outside the registry entirely) — recompiling
        // without resetting this would make buildBarriers() think "swapchain" is already in
        // COLOR_ATTACHMENT_OPTIMAL and skip the barrier that puts it there for real. FrameGraph::resize()
        // avoids this the same way, via ResourceRegistry::rebuild() resetting every external image.
        viewer.resources().setImageLayout("swapchain", VK_IMAGE_LAYOUT_UNDEFINED);

        geometryPass.rebuild(viewer.frameGraph(), gpuMeshLayout, {
            .cameraBufferResourceName = cameraUploader.bufferName(),
            .vertexBufferResourceNames = { {0, mainMeshPositionBufferName}, {1, mainMeshVertexBufferName} },
            .vertexBufferUploadResult = meshPositions,
            .indexBufferUploadResult = indexBuffer,
            .meshTransforms = currentMeshTransforms(),
            .indexBufferResourceName = mainMeshIndexBufferName,
            .faceGroupBufferResourceName = mainMeshFaceGroupBufferName,
            .diffuseTextureArrayResourceName = material.textureNameMap.at(config.diffuseTextureName),
            .normalTextureArrayResourceName = material.textureNameMap.at(config.normalTextureName),
            .metallicRoughnessTextureArrayResourceName = material.textureNameMap.at(config.metallicRoughnessTextureName),
            .emissiveTextureArrayResourceName = material.textureNameMap.at(config.emissiveTextureName),
            .materialBufferResourceName = material.materialInfoBufferName,
            .materialCount = static_cast<uint32_t>(combinedMaterials.size()),
        });

        viewer.frameGraph().compile();
    };

    // Recomputes every extraModelObject's absolute material offset from its pristine
    // localFaceGroups and re-bakes it into the live Mesh. Only needed when lightVisualObjects
    // grows — that block sits before extraModelObjects in buildCombinedMaterials(), so a new
    // light's material shifts every already-loaded mesh's offset out from under it. Call this
    // before rebuildSceneGeometry() whenever a light is added.
    auto rebakeExtraModelFaceGroups = [&]() {
        uint32_t base = lightVisualMaterialBase + static_cast<uint32_t>(lightVisualObjects.size());
        for (auto &entry : extraModelObjects)
        {
            lr::Mesh &mesh = entry.object->getComponent<lr::StaticMesh>().mesh();
            for (size_t f = 0; f < mesh.faceGroups.size(); ++f)
                mesh.faceGroups[f] = entry.localFaceGroups[f] + base;
            base += static_cast<uint32_t>(entry.object->getComponent<lr::StaticMesh>().materials().size());
        }
    };

    // Adds a Point light to the scene at runtime. Every light — not just Area lights — carries a
    // hidden quad visual (see the startup light-visual loop above) so it can be switched to Area
    // type later via the Light component's type combo without needing to regrow GPU buffers; a
    // freshly added Point light needs that same slot pre-allocated.
    auto addPointLightObject = [&]() {
        lr::PointLight point;
        point.color = glm::vec3(1.0f);
        point.intensity = 5.0f;

        lr::SceneObject *lightObject = sceneObjects.emplace_back(std::make_unique<lr::SceneObject>()).get();
        lightObject->addComponent<lr::Transform>(
            glm::vec3(1.5f * static_cast<float>(lightVisualObjects.size()), 2.0f, 0.0f));
        lightObject->addComponent<lr::Light>(lr::LightVariant{point});
        lightObject->name = "Point Light " + std::to_string(lightVisualObjects.size() + 1);

        const uint32_t materialIndex = lightVisualMaterialBase + static_cast<uint32_t>(lightVisualObjects.size());
        lr::Mesh quadMesh;
        lr::buildAreaLightQuadMesh(quadMesh, lightObject->getComponent<lr::Transform>(),
                                    hiddenAreaLightVisual, materialIndex, areaLightVisualConfig);
        std::vector<lr::Material> quadMaterials;
        quadMaterials.push_back(lr::buildAreaLightMaterial(hiddenAreaLightVisual, areaLightVisualConfig));
        lightObject->addComponent<lr::StaticMesh>(quadMesh, quadMaterials, /*hideFromGui=*/true);

        lightVisualObjects.push_back(lightObject);
        wireLightObject(lightObject);

        rebakeExtraModelFaceGroups();
        updateLightList();
        rebuildSceneGeometry();
    };

    // Loads a mesh from disk (picked via the file dialog below) and adds it to the scene at
    // runtime — same pattern as the original "Load Another Model" proof of concept, generalized
    // to a caller-supplied path (dispatched to GltfLoader or ObjLoader by extension, both
    // producing materials keyed to match gpuMaterialLayout — see objConfig above) and keeping a
    // pristine copy of the face groups so rebakeExtraModelFaceGroups can correct this object's
    // material offset later if a light gets added after it.
    auto addMeshFromFile = [&](const std::filesystem::path &path) {
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });

        lr::Mesh newMesh;
        std::vector<lr::Material> newMaterials;

        if (ext == ".obj")
        {
            try
            {
                lr::ObjMeshLoadResult result = objLoader.load(path, objConfig);
                newMesh = std::move(result.mesh);
                newMaterials = std::move(result.materials);
            }
            catch (const std::exception &e)
            {
                spdlog::error("Add Mesh: ObjLoader failed for '{}': {}", path.string(), e.what());
                return;
            }
        }
        else
        {
            auto [newSequence, gltfMaterials] = gltfLoader.load(path, config);
            if (newSequence.empty())
            {
                spdlog::error("Add Mesh: GltfLoader returned empty sequence for '{}'", path.string());
                return;
            }
            newMesh = std::move(newSequence.frames.front());
            newMaterials = std::move(gltfMaterials);
        }

        std::vector<uint32_t> localFaceGroups = newMesh.faceGroups;
        const uint32_t materialBase = static_cast<uint32_t>(buildCombinedMaterials().size());
        for (auto &group : newMesh.faceGroups)
            group += materialBase;

        lr::SceneObject *newObject = sceneObjects.emplace_back(std::make_unique<lr::SceneObject>()).get();
        newObject->addComponent<lr::Transform>(
            glm::vec3(2.5f * static_cast<float>(extraModelObjects.size() + 1), 0.0f, 0.0f));
        newObject->addComponent<lr::StaticMesh>(newMesh, newMaterials);
        newObject->name = path.stem().string();

        extraModelObjects.push_back({ newObject, std::move(localFaceGroups) });
        rebuildSceneGeometry();
    };

    lr::AmbientOcclusionPass aoPass({
        .cameraBufferResourceName = cameraUploader.bufferName(),
    });
    aoPass.uploadResources(viewer.resources());
    aoPass.build(viewer.frameGraph());

    lr::PbrPass pbrPass({
        .cameraBufferResourceName = cameraUploader.bufferName(),
        .lightBufferResourceName = lightUploader.bufferName(),
        .numLights = &lightUploader.numLights(),
        .pfMips = 8,
    });
    pbrPass.uploadResources(viewer.resources());
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

    // Vertex-selection points only apply to the main mesh, not the area light quads — meshPositions
    // now covers both (see geometryMeshes above), so scope this pass to just its first entry.
    const lr::VertexBufferUploadResult mainMeshPositionResult{
        .singleMeshResults = { meshPositions.singleMeshResults.front() }
    };

    lr::OverlayPointsPass overlayPointsPass({
        .cameraBufferResourceName   = cameraUploader.bufferName(),
        .positionBufferResourceName = mainMeshPositionBufferName,
        .colorBufferResourceName    = mainMeshColorBufferName,
        .positionBufferUploadResult = mainMeshPositionResult,
        .vertexCounts               = { staticMesh.mesh().vertexCount() },
        .meshTransform              = &meshObject->getComponent<lr::Transform>(),
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
        // Must repack the full combined mesh list (main mesh + area light quads), not just the
        // static mesh — mainMeshPositionBufferName is sized for all of them (see geometryMeshes
        // above), and reuploadBuffer's staging buffer needs to match that size.
        meshUploader.updateVertexBuffer(geometryMeshes, meshPositionUploadConfig);
    });

    lr::CommandManager commandManager;

    lr::SelectionManager selectionManager(staticMesh.mesh().positions, meshObject->getComponent<lr::Transform>(), viewer.input());
    selectionManager.setSelectTool(std::make_unique<lr::BoxSelectionTool>(viewer.input(), *camera, selectionManager));
    selectionManager.registerHighlightChangedCallback([&]() {
        std::fill(pointColors.begin(), pointColors.end(), glm::vec3(1.0f, 0.0f, 1.0f));
        for (uint32_t idx : selectionManager.getHighlightedIndices())
            pointColors[idx] = glm::vec3(1.0f, 0.8f, 0.0f);  // orange = selected
        viewer.resources().updateBuffer(
            mainMeshColorBufferName,
            pointColors.data(),
            pointColors.size() * sizeof(glm::vec3));
    });

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

    // Lets the user grow the scene at runtime from a small predetermined list: a Point light
    // (addPointLightObject) or a mesh picked from disk via ImGuiFileDialog (addMeshFromFile).
    viewer.onGui([&]() {
        ImGui::Begin("Add Object");

        if (ImGui::Button("Add Point Light"))
            addPointLightObject();

        if (ImGui::Button("Add Mesh..."))
            ImGuiFileDialog::Instance()->OpenDialog(
                "AddMeshDialog", "Choose Mesh", ".gltf,.glb,.obj{.gltf,.glb,.obj}", IGFD::FileDialogConfig{});

        if (ImGuiFileDialog::Instance()->Display("AddMeshDialog"))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
                addMeshFromFile(ImGuiFileDialog::Instance()->GetFilePathName());
            ImGuiFileDialog::Instance()->Close();
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

    viewer.run();
    return 0;
}
catch (const std::exception &e)
{
    spdlog::error("Fatal: {}", e.what());
    throw;
    return 1;
}
