#include "core/app/Viewer.hpp"
#include "core/Paths.hpp"
#include "core/loaders/Material.hpp"
#include "core/loaders/MaterialStore.hpp"
#include "core/overlay/OverlayMesh.hpp"
#include "core/passes/final/FinalPass.hpp"
#include "core/passes/geometry/GeometryPass.hpp"
#include "core/passes/heatmap/HeatmapPass.hpp"
#include "core/passes/ibl/IblPass.hpp"
#include "core/passes/pbr/PbrPass.hpp"
#include "core/passes/ambientocclusion/AmbientOcclusionPass.hpp"
#include "core/passes/overlaygeometry/OverlayGeometryPass.hpp"
#include "core/passes/overlaylines/OverlayLinesPass.hpp"
#include "core/passes/overlaypoints/OverlayPointsPass.hpp"
#include "core/framegraph/ImageReadback.hpp"

#include "core/scene/AreaLightVisual.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/editor/camera/SphericalCameraController.hpp"
#include "core/editor/gizmo/GizmoManager.hpp"
#include "core/editor/gizmo/TranslateDragHandlerGizmo.hpp"
#include "core/editor/gizmo/translate/TranslateArrowGizmo.hpp"
#include "core/editor/gizmo/translate/TranslateBoxGizmo.hpp"
#include "core/editor/SceneObjectDragHandler.hpp"
#include "core/editor/selection/BoxSelectionTool.hpp"
#include "core/editor/selection/SelectionManager.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/editor/DefaultVertexDragHandler.hpp"
#include "core/scene/SceneManager.hpp"
#include "core/scene/Scene.hpp"

#include "features/arap/ArapTool.hpp"
#include "features/laplace_beltrami/LaplaceBeltramiTool.hpp"
#include "features/rigid_body/ColliderComponent.hpp"
#include "features/rigid_body/ColliderVisual.hpp"
#include "features/rigid_body/PhysicsWorld.hpp"
#include "features/rigid_body/RigidBodyComponent.hpp"

#include <ImGuiFileDialog.h>
#include <imgui.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec4.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

int main()
try
{
    spdlog::set_level(spdlog::level::debug);

    lr::Viewer viewer({.title = "lr"});

    namespace fs = std::filesystem;

    // -------------------------------------------------------------------------
    // IBL preprocessing  (runs once before the frame loop)
    // -------------------------------------------------------------------------

    const lr::IBLPass::Config iblConfig{
        .envRes = 2048,
        .irrRes = 32,
        .pfRes  = 2048,
        .pfMips = 8,
    };

    {
        lr::IBLPass iblPass(iblConfig);
        iblPass.uploadResources(viewer.resources());
        lr::FrameGraph iblGraph(viewer.context(), viewer.resources());
        iblPass.build(iblGraph);
        iblGraph.executeAndWait({
            {"ibl_irradiance", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {"ibl_prefiltered", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        });
    }

    // -------------------------------------------------------------------------
    // Scene setup
    // -------------------------------------------------------------------------

    lr::Scene scene;

    lr::GltfLoaderConfig config{
        .normalAttributeName          = "normal",
        .tangentAttributeName         = "tangent",
        .uvAttributeName              = "uv",
        .diffuseTextureName           = "baseColorTexture",
        .normalTextureName            = "normalTexture",
        .metallicRoughnessTextureName = "metallicRoughnessTexture",
        .emissiveTextureName          = "emissiveTexture",
        .baseDiffuseName              = "baseDiffuse",
        .baseRoughnessName            = "baseRoughness",
        .baseMetallicName             = "baseMetallic",
        .baseEmissiveName             = "baseEmissive",
    };
    const lr::SceneLoaderConfig sceneLoadConfig{
        .gltf = config,
        .obj = {
            .normalAttributeName  = config.normalAttributeName,
            .tangentAttributeName = config.tangentAttributeName,
            .uvAttributeName      = config.uvAttributeName,
            .diffuseTextureName   = config.diffuseTextureName,
            .normalTextureName    = config.normalTextureName,
            .roughnessTextureName = config.metallicRoughnessTextureName,
            .emissiveTextureName  = config.emissiveTextureName,
            .baseDiffuseName      = config.baseDiffuseName,
            .baseRoughnessName    = config.baseRoughnessName,
            .baseMetallicName     = config.baseMetallicName,
            .baseEmissiveName     = config.baseEmissiveName,
        },
    };

    // Flat, up-front reservation for the MaterialStore's GPU-side buffer/texture-array capacity —
    // growing this would mean rebuilding the frame graph's descriptor sets (see MaterialStore.hpp),
    // so it's a generous constant rather than something computed tightly from scene content.
    constexpr uint32_t kMaterialCapacity = 256;

    lr::SceneManager sceneManager(viewer.resources(), kMaterialCapacity, [config]() {
        lr::Material material;
        material.name                                 = "Unused Material Slot";
        material.parameters[config.baseDiffuseName]   = lr::MaterialParam::ColorRGBA{glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)};
        material.parameters[config.baseEmissiveName]  = lr::MaterialParam::ColorRGB{glm::vec3(0.0f)};
        material.parameters[config.baseRoughnessName] = lr::MaterialParam::NormalizedFloat{1.0f};
        material.parameters[config.baseMetallicName]  = lr::MaterialParam::NormalizedFloat{0.0f};
        return material;
    });
    sceneManager.setScene(scene);

    lr::SceneObject *camera = &scene.createSceneObject();
    camera->addComponent<lr::Camera>();
    camera->addComponent<lr::TransformComponent>();
    camera->name = "Main Camera";
    sceneManager.setDefaultCamera(*camera);
    scene.protectSceneObject(camera->id());

    // LIGHT
    {
        lr::DirectionalLight light;
        light.color     = glm::vec3(1.0f, 1.0f, 1.0f);
        light.intensity = 1.0f;

        lr::SceneObject &lightObject = scene.createSceneObject();
        lightObject.addComponent<lr::TransformComponent>();
        lightObject.addComponent<lr::Light>(light);
        lightObject.name = "Light";
    }

    // MESH
    const fs::path meshPath = lr::paths::assetDir / "samples/models/lion_head_4k.glb";

    sceneManager.load(meshPath, sceneLoadConfig);

    // LIGHT VISUALS — every light, not just ones that start out as AreaLight, gets its own MeshComponent
    // (a quad), separate from scene geometry. The quad still draws through the
    // same GeometryPass (see AreaLightVisual.hpp for why the visual needs to be real
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

    lr::SceneObject *meshObject = &sceneManager.selectedMeshObject();
    auto &meshComponent = meshObject->getComponent<lr::MeshComponent>();

    // Also read by the AmbientOcclusionPass config below, to size sphereRadius off the model.
    glm::vec3 boundsMin(0.0f);
    glm::vec3 boundsMax(0.0f);

    // Seed the demo object with a local-space box collider fitted to its mesh. Collision
    // detection will consume the same component later; for now this also makes the collider
    // inspector and visualization immediately available in the sample application.
    if (!meshComponent.mesh().positions().empty())
    {
        boundsMin = meshComponent.mesh().positions().front();
        boundsMax = boundsMin;
        for (const glm::vec3 &position : meshComponent.mesh().positions())
        {
            boundsMin = glm::min(boundsMin, position);
            boundsMax = glm::max(boundsMax, position);
        }

        lr::Collider collider;
        collider.shape         = lr::BoxCollider{glm::max((boundsMax - boundsMin) * 0.5f, glm::vec3(0.001f))};
        collider.localPosition = (boundsMin + boundsMax) * 0.5f;
        meshObject->addComponent<lr::ColliderComponent>(std::move(collider));
        meshObject->addComponent<lr::RigidBodyComponent>();

        // TEST OBJECT — a free-floating sphere collider hovering above the lion, for exercising
        // collision detection against the mesh's box collider once that lands.
        constexpr float kTestSphereRadius = 0.3f;
        lr::SceneObject &testSphereObject = scene.createSceneObject();
        testSphereObject.name             = "Test Sphere";
        testSphereObject.addComponent<lr::TransformComponent>(
            glm::vec3((boundsMin.x + boundsMax.x) * 0.5f, boundsMax.y + kTestSphereRadius * 4.0f,
                     (boundsMin.z + boundsMax.z) * 0.5f));

        lr::Collider testSphereCollider;
        testSphereCollider.shape = lr::SphereCollider{kTestSphereRadius};
        testSphereObject.addComponent<lr::ColliderComponent>(std::move(testSphereCollider));
        testSphereObject.addComponent<lr::RigidBodyComponent>();

        // TEST OBJECT — a static ground plane collider below the lion, for exercising collision
        // detection against the mesh's box collider once that lands.
        lr::SceneObject &testPlaneObject = scene.createSceneObject();
        testPlaneObject.name             = "Test Ground Plane";
        testPlaneObject.addComponent<lr::TransformComponent>(
            glm::vec3((boundsMin.x + boundsMax.x) * 0.5f, boundsMin.y, (boundsMin.z + boundsMax.z) * 0.5f));

        lr::Collider testPlaneCollider;
        const glm::vec3 extents = boundsMax - boundsMin;
        testPlaneCollider.shape = lr::PlaneCollider{
            .offset      = 0.0f,
            .halfExtents = glm::vec2(std::max(extents.x, extents.z), std::max(extents.x, extents.z)),
        };
        testPlaneObject.addComponent<lr::ColliderComponent>(std::move(testPlaneCollider));
        testPlaneObject.addComponent<lr::RigidBodyComponent>(1.0f, lr::RigidBodyType::Static);

        // CORNELL BOX — floor, ceiling, back, left (red) and right (green) walls built from
        // scratch and sized around the lion's bounds. The floor sits exactly at boundsMin.y, so
        // the lion's bottom (already there) touches the box floor with no extra transform. The
        // front is left open so the camera can see inside.
        {
            const glm::vec3 center     = (boundsMin + boundsMax) * 0.5f;
            const glm::vec3 extents    = boundsMax - boundsMin;
            const float     halfX      = std::max(extents.x, 0.001f) * 1.5f;
            const float     halfZ      = std::max(extents.z, 0.001f) * 1.5f;
            const float     floorY     = boundsMin.y;
            const float     ceilingY   = boundsMin.y + std::max(extents.y, 0.001f) * 2.5f;
            const float     halfY      = (ceilingY - floorY) * 0.5f;
            const glm::vec3 boxCenter  = glm::vec3(center.x, (floorY + ceilingY) * 0.5f, center.z);

            // Each entry: world-space center, right axis, up axis (their cross product is the
            // quad's facing normal — see AreaLightVisual.cpp for the same winding convention),
            // and the material color. Shared by the room walls (inward-facing) and the pedestal
            // cube below (outward-facing) — same quad-building code either way.
            struct QuadSpec
            {
                std::string name;
                glm::vec3   center;
                glm::vec3   right;
                glm::vec3   up;
                glm::vec3   normal;
                glm::vec3   color;
            };

            const auto buildQuadObject = [&](const QuadSpec &quad) {
                lr::Material material;
                material.name                                 = quad.name;
                material.parameters[config.baseDiffuseName]   = lr::MaterialParam::ColorRGBA{glm::vec4(quad.color, 1.0f)};
                material.parameters[config.baseEmissiveName]  = lr::MaterialParam::ColorRGB{glm::vec3(0.0f)};
                material.parameters[config.baseRoughnessName] = lr::MaterialParam::NormalizedFloat{1.0f};
                material.parameters[config.baseMetallicName]  = lr::MaterialParam::NormalizedFloat{0.0f};
                const lr::MaterialHandle handle = sceneManager.materialStore().acquire(std::move(material));

                std::vector<glm::vec3> positions = {
                    quad.center - quad.right - quad.up,
                    quad.center + quad.right - quad.up,
                    quad.center + quad.right + quad.up,
                    quad.center - quad.right + quad.up,
                };
                std::vector<uint32_t>   positionIndices = {0, 1, 2, 3};
                std::vector<glm::uvec3> faces           = {{0, 1, 2}, {0, 2, 3}};

                lr::Mesh quadMesh;
                quadMesh.setTopology(std::move(positions), std::move(positionIndices), std::move(faces));
                quadMesh.setPerVertexArray<glm::vec3>(config.normalAttributeName, std::vector<glm::vec3>(4, quad.normal));
                quadMesh.setPerVertexArray<glm::vec4>(
                    config.tangentAttributeName,
                    std::vector<glm::vec4>(4, glm::vec4(glm::normalize(quad.right), 1.0f)));
                const std::vector<glm::vec2> uvs = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
                quadMesh.setPerVertexArray<glm::vec2>(config.uvAttributeName, uvs);
                quadMesh.setFaceGroups({handle, handle});

                const lr::MeshHandle meshHandle = sceneManager.meshStore().add(std::move(quadMesh));

                lr::SceneObject &quadObject = scene.createSceneObject();
                quadObject.name             = quad.name;
                quadObject.addComponent<lr::TransformComponent>();
                quadObject.addComponent<lr::MeshComponent>(meshHandle, sceneManager.meshStore(),
                                                            std::vector<lr::MaterialHandle>{handle},
                                                            sceneManager.materialStore());
                sceneManager.addMeshObject(quadObject);
            };

            const std::vector<QuadSpec> walls = {
                {"Cornell Floor", {boxCenter.x, floorY, boxCenter.z}, {0, 0, halfZ}, {halfX, 0, 0}, {0, 1, 0},
                 glm::vec3(0.73f)},
                {"Cornell Ceiling", {boxCenter.x, ceilingY, boxCenter.z}, {halfX, 0, 0}, {0, 0, halfZ}, {0, -1, 0},
                 glm::vec3(0.73f)},
                {"Cornell Back Wall", {boxCenter.x, boxCenter.y, center.z - halfZ}, {halfX, 0, 0}, {0, halfY, 0},
                 {0, 0, 1}, glm::vec3(0.73f)},
                {"Cornell Left Wall", {center.x - halfX, boxCenter.y, boxCenter.z}, {0, halfY, 0}, {0, 0, halfZ},
                 {1, 0, 0}, glm::vec3(0.75f, 0.05f, 0.05f)},
                {"Cornell Right Wall", {center.x + halfX, boxCenter.y, boxCenter.z}, {0, 0, halfZ}, {0, halfY, 0},
                 {-1, 0, 0}, glm::vec3(0.05f, 0.6f, 0.05f)},
            };
            for (const QuadSpec &wall : walls)
            {
                buildQuadObject(wall);
            }

            // PEDESTAL CUBE — a clean, sharp-edged procedural cube standing in for the lion's own
            // sculpted base (which has a faceted/beveled edge — a diagnostic for the HBAO dotted-
            // edge artifact: if a hand-built, single-flat-face-per-side cube shows the same dots
            // along its edges, the artifact isn't specific to the lion's faceted geometry after
            // all). The lion is lifted so its own (unmodified) base now sits on top of this cube
            // rather than on the Cornell floor directly.
            {
                const float     pedestalHalf   = std::max(std::max(extents.x, extents.z) * 0.55f, 0.001f);
                const float     pedestalHeight = std::max(extents.y, 0.001f) * 0.35f;
                const glm::vec3 pedestalCenter(center.x, floorY + pedestalHeight * 0.5f, center.z);
                const float     px = pedestalHalf;
                const float     py = pedestalHeight * 0.5f;
                const float     pz = pedestalHalf;

                const std::vector<QuadSpec> pedestalFaces = {
                    {"Pedestal Top", pedestalCenter + glm::vec3(0, py, 0), {0, 0, pz}, {px, 0, 0}, {0, 1, 0},
                     glm::vec3(0.73f)},
                    {"Pedestal Bottom", pedestalCenter - glm::vec3(0, py, 0), {px, 0, 0}, {0, 0, pz}, {0, -1, 0},
                     glm::vec3(0.73f)},
                    {"Pedestal Front", pedestalCenter + glm::vec3(0, 0, pz), {px, 0, 0}, {0, py, 0}, {0, 0, 1},
                     glm::vec3(0.73f)},
                    {"Pedestal Back", pedestalCenter - glm::vec3(0, 0, pz), {0, py, 0}, {px, 0, 0}, {0, 0, -1},
                     glm::vec3(0.73f)},
                    {"Pedestal Right", pedestalCenter + glm::vec3(px, 0, 0), {0, py, 0}, {0, 0, pz}, {1, 0, 0},
                     glm::vec3(0.73f)},
                    {"Pedestal Left", pedestalCenter - glm::vec3(px, 0, 0), {0, 0, pz}, {0, py, 0}, {-1, 0, 0},
                     glm::vec3(0.73f)},
                };
                for (const QuadSpec &face : pedestalFaces)
                {
                    buildQuadObject(face);
                }

                // Lift the lion so its own (local-space, unchanged) bottom lands on the pedestal's
                // top face instead of the Cornell floor.
                lr::TransformComponent &meshTransform = meshObject->getComponent<lr::TransformComponent>();
                meshTransform.setPosition(meshTransform.transform().position() + glm::vec3(0.0f, pedestalHeight, 0.0f));
            }
        }
    }

    lr::PhysicsWorld physicsWorld(scene);

    // -------------------------------------------------------------------------
    // Resource uploads
    // -------------------------------------------------------------------------

    // This matches the expected layout in geometry.frag
    lr::GpuMaterialLayout gpuMaterialLayout;
    gpuMaterialLayout.setStride(48)
        .addScalar(config.baseDiffuseName, 0, sizeof(glm::vec4))
        .addScalar(config.baseEmissiveName, 16, sizeof(glm::vec3))
        .addScalar(config.baseRoughnessName, 32, sizeof(float))
        .addScalar(config.baseMetallicName, 36, sizeof(float))
        .addTexture(config.diffuseTextureName, VK_FORMAT_R8G8B8A8_SRGB)
        .addTexture(config.normalTextureName, VK_FORMAT_R8G8B8A8_UNORM)
        .addTexture(config.metallicRoughnessTextureName, VK_FORMAT_R8G8B8A8_UNORM)
        .addTexture(config.emissiveTextureName, VK_FORMAT_R8G8B8A8_SRGB);

    // Builds light visuals, uploads the initial lights/mesh/material/camera buffers, and wires the
    // change listeners that keep the camera UBO and materials SSBO in sync afterward —
    // see SceneManager::initialize().
    sceneManager.initialize(areaLightVisualConfig, gpuMaterialLayout,
                            {config.normalAttributeName, config.tangentAttributeName, config.uvAttributeName},
                            viewer.input());

    // -------------------------------------------------------------------------
    // Frame graph passes
    // -------------------------------------------------------------------------

    const VkFormat swapchainFormat = viewer.frameGraph().resources().getImage("swapchain")->format;

    lr::GeometryPass  geometryPass({
         .cameraBufferResourceName    = sceneManager.cameraBufferName(),
         .vertexBufferResourceNames   = {{0, sceneManager.meshPositionBufferName()},
                                         {1, sceneManager.meshVertexBufferName()}},
         .vertexBufferUploadResult    = sceneManager.meshPositions(),
         .indexBufferUploadResult     = sceneManager.indexBuffer(),
         .meshTransforms              = sceneManager.meshTransforms(),
         .meshObjects                 = sceneManager.geometryObjects(),
         .skinDrawInfos               = sceneManager.skinUploadResult().drawInfos,
         .indexBufferResourceName     = sceneManager.meshIndexBufferName(),
         .faceGroupBufferResourceName = sceneManager.meshFaceGroupBufferName(),
         .diffuseTextureArrayResourceName =
            sceneManager.materialUploadResult().textureNameMap.at(config.diffuseTextureName),
         .normalTextureArrayResourceName =
            sceneManager.materialUploadResult().textureNameMap.at(config.normalTextureName),
         .metallicRoughnessTextureArrayResourceName =
            sceneManager.materialUploadResult().textureNameMap.at(config.metallicRoughnessTextureName),
         .emissiveTextureArrayResourceName =
            sceneManager.materialUploadResult().textureNameMap.at(config.emissiveTextureName),
         .materialBufferResourceName = sceneManager.materialUploadResult().materialInfoBufferName,
         .skinInfluenceEntriesBufferResourceName = sceneManager.skinInfluenceEntriesBufferName(),
         .skinInfluenceOffsetsBufferResourceName = sceneManager.skinInfluenceOffsetsBufferName(),
         .skinPositionIndicesBufferResourceName = sceneManager.skinPositionIndicesBufferName(),
         .skinJointMatricesBufferResourceName = sceneManager.skinJointMatricesBufferName(),

         .materialCount = sceneManager.materialStore().capacity(),
    });
    lr::GpuMeshLayout gpuMeshLayout(meshComponent.mesh().layout());

    gpuMeshLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT);
    gpuMeshLayout.map(config.normalAttributeName, 1, 1, VK_FORMAT_R32G32B32_SFLOAT);
    gpuMeshLayout.map(config.tangentAttributeName, 1, 2, VK_FORMAT_R32G32B32A32_SFLOAT);
    gpuMeshLayout.map(config.uvAttributeName, 1, 3, VK_FORMAT_R32G32_SFLOAT);

    geometryPass.build(viewer.frameGraph(), gpuMeshLayout);

    lr::HeatmapPass heatmapPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
        .vertexBufferResourceName = sceneManager.selectedMeshHeatmapBufferName(),
        .indexBufferResourceName  = sceneManager.meshIndexBufferName(),
        .vertexBufferUploadResult = sceneManager.selectedMeshHeatmap(),
        .indexBufferUploadResult  = sceneManager.indexBuffer(),
        .meshTransform            = &meshObject->getComponent<lr::TransformComponent>(),
    });

    lr::GpuMeshLayout heatmapMeshLayout(meshComponent.mesh().layout());
    heatmapMeshLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT);
    heatmapMeshLayout.map("heatmapColors", 0, 1, VK_FORMAT_R32G32B32_SFLOAT);

    heatmapPass.build(viewer.frameGraph(), heatmapMeshLayout);

    // sphereRadius is 2% of the lion's largest local-space extent (see boundsMin/boundsMax above).
    const glm::vec3 lionExtents        = boundsMax - boundsMin;
    const float     lionLargestExtent  = std::max({lionExtents.x, lionExtents.y, lionExtents.z});

    lr::AmbientOcclusionPass aoPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
        .sphereRadius             = lionLargestExtent * 0.02f,
    });
    aoPass.uploadResources(viewer.resources());
    aoPass.build(viewer.frameGraph());

    lr::PbrPass pbrPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
        .lightBufferResourceName  = sceneManager.lightBufferName(),
        .numLights                = sceneManager.numLights(),
        .pfMips                   = 8,
    });
    pbrPass.uploadResources(viewer.resources());
    pbrPass.build(viewer.frameGraph());

    lr::OverlayGeometryPass overlayGeometryPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
    });
    overlayGeometryPass.uploadResources(viewer.resources());
    overlayGeometryPass.build(viewer.frameGraph());
    overlayGeometryPass.setInstances({});

    lr::OverlayLinesPass overlayLinesPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
    }, viewer.resources());
    overlayLinesPass.build(viewer.frameGraph());
    overlayLinesPass.setLines({});

    lr::GpuMeshLayout pointsMeshLayout(meshComponent.mesh().layout());
    pointsMeshLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT);
    pointsMeshLayout.mapUniqueVertex("color", 0, 1, VK_FORMAT_R32G32B32_SFLOAT);

    lr::OverlayPointsPass overlayPointsPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
        .pointsBufferResourceName = sceneManager.selectedMeshPointsBufferName(),
        .pointsBufferUploadResult = sceneManager.selectedMeshPoints(),
        .vertexCounts             = {meshComponent.mesh().uniquePositionCount()},
        .meshTransform            = &meshObject->getComponent<lr::TransformComponent>(),
    });
    overlayPointsPass.build(viewer.frameGraph(), pointsMeshLayout);

    lr::FinalPass finalPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
        .swapchainFormat          = swapchainFormat,
    });
    finalPass.build(viewer.frameGraph());

    // -------------------------------------------------------------------------
    // Editor state — vertex picking, selection and gizmo managers
    // -------------------------------------------------------------------------

    const auto applyEditorMode = [&](lr::EditorMode mode) {
        geometryPass.setSkinningEnabled(mode == lr::EditorMode::View);
        overlayPointsPass.setEnabled(mode == lr::EditorMode::Edit);
        heatmapPass.setEnabled(mode == lr::EditorMode::Analysis);
    };
    sceneManager.registerEditorModeChangedCallback(applyEditorMode);
    applyEditorMode(sceneManager.editorMode());

    // Gizmo hover — reads the picking image from the previous frame
    lr::ImageReadback gizmoReadback(viewer.context(), viewer.allocator());

    lr::VertexManager vertexManager(meshComponent.mesh());
    vertexManager.registerUpdateCallback([&]() {
        sceneManager.updateSelectedMeshPositions();
    });

    lr::CommandManager commandManager;

    // SceneManager owns the SelectionManager (constructed for the initial selected mesh, see
    // SceneManager::selectionManager()) since it needs to wire selection-highlight changes straight
    // to the GPU color buffer; this is just a local alias to keep the call sites below unchanged.
    lr::SelectionManager &selectionManager = sceneManager.selectionManager();
    selectionManager.setSelectTool(std::make_unique<lr::BoxSelectionTool>(viewer.input(), *camera, selectionManager));

    // What the translate gizmos drive by default (plain vertex-drag editing). ArapTool swaps
    // this out for an ARAP-solve handler on the same gizmo instances once a precompute succeeds.
    lr::DefaultVertexDragHandler defaultHandler(vertexManager, selectionManager, commandManager);
    lr::SceneObjectDragHandler   sceneObjectHandler(commandManager);

    lr::GizmoManager gizmoManager(overlayGeometryPass, viewer.input());

    auto arrowXGizmo = std::make_unique<lr::TranslateArrowGizmo>(lr::TranslateArrowGizmoAxis::X, *camera,
                                                                 viewer.input(), defaultHandler);
    auto arrowYGizmo = std::make_unique<lr::TranslateArrowGizmo>(lr::TranslateArrowGizmoAxis::Y, *camera,
                                                                 viewer.input(), defaultHandler);
    auto arrowZGizmo = std::make_unique<lr::TranslateArrowGizmo>(lr::TranslateArrowGizmoAxis::Z, *camera,
                                                                 viewer.input(), defaultHandler);
    auto boxGizmo    = std::make_unique<lr::TranslateBoxGizmo>(*camera, viewer.input(), defaultHandler);

    // Raw pointers kept for ArapTool (needs a generic TranslateDragHandlerGizmo list) and the gizmo-
    // positioning loop below (reads whichever handler is currently wired) — ownership moves to
    // gizmoManager via addGizmo() just below.
    lr::TranslateArrowGizmo *arrowX      = arrowXGizmo.get();
    lr::TranslateArrowGizmo *arrowY      = arrowYGizmo.get();
    lr::TranslateArrowGizmo *arrowZ      = arrowZGizmo.get();
    lr::TranslateBoxGizmo   *boxGizmoPtr = boxGizmo.get();

    const std::vector<int> translateGizmoIds = {
        gizmoManager.addGizmo(std::move(arrowXGizmo)),
        gizmoManager.addGizmo(std::move(arrowYGizmo)),
        gizmoManager.addGizmo(std::move(arrowZGizmo)),
        gizmoManager.addGizmo(std::move(boxGizmo)),
    };
    for (int id : translateGizmoIds)
    {
        gizmoManager.hideGizmo(id);
    }

    const std::vector<lr::TranslateDragHandlerGizmo *> dragHandlerGizmos = {arrowX, arrowY, arrowZ, boxGizmoPtr};
    lr::TranslateDragHandler *objectTranslateReturnHandler = &defaultHandler;

    enum class ObjectTransformTool
    {
        None,
        Translate,
    };
    ObjectTransformTool objectTransformTool       = ObjectTransformTool::None;
    bool                objectTransformWindowOpen = false;

    lr::ArapTool arapTool(selectionManager, vertexManager, commandManager, meshComponent.mesh(), defaultHandler,
                          dragHandlerGizmos);

    lr::LaplaceBeltramiTool laplaceBeltramiTool(meshComponent.mesh(), sceneManager, vertexManager);

    // Ties vertex editing (SelectionManager, VertexManager, ArapTool, the vertex-picking points
    // overlay) to whatever's selected in the Scene Hierarchy. ARAP, Laplace-Beltrami analysis,
    // picking, and both selected-mesh GPU overlays are all
    // rebound together so every editing/analysis operation addresses the same object.
    scene.registerSelectionChangedCallback([&](lr::SceneObjectId id) {
        lr::SceneObject &object = scene.getSceneObject(id);
        objectTransformTool       = ObjectTransformTool::None;
        objectTransformWindowOpen = true;
        sceneObjectHandler.setTarget(object.hasComponent<lr::TransformComponent>() ? &object : nullptr);
        if (!lr::SceneManager::isEditable(object))
        {
            if (sceneManager.editorMode() == lr::EditorMode::Edit)
            {
                sceneManager.setEditorMode(lr::EditorMode::View);
            }
            return;
        }
        if (sceneManager.editedMeshObject() == &object)
        {
            return;
        }

        // setEditedMeshObject() replaces the points-picking buffer in place (see
        // ResourceRegistry::replaceUploadedBuffer) — matches the wait-then-replace pattern the
        // HDRI reload path already uses for the same reason (in-flight frames may still read it).
        viewer.context().waitIdle();
        sceneManager.setEditedMeshObject(object);

        lr::Mesh &mesh = object.getComponent<lr::MeshComponent>().mesh();
        vertexManager.rebind(mesh);
        arapTool.rebind(mesh);
        laplaceBeltramiTool.rebind(mesh);
        heatmapPass.setMeshSource(sceneManager.selectedMeshHeatmap(), sceneManager.selectedMeshIndexRange(),
                                  object.getComponent<lr::TransformComponent>());
        overlayPointsPass.setPointsSource(sceneManager.selectedMeshPoints(), mesh.uniquePositionCount(),
                                          object.getComponent<lr::TransformComponent>());
    });

    scene.registerObjectsDestroyedCallback([&](std::span<const lr::SceneObjectId> ids) {
        viewer.context().waitIdle();
        if (sceneObjectHandler.target() &&
            std::ranges::find(ids, sceneObjectHandler.target()->id()) != ids.end())
        {
            sceneObjectHandler.setTarget(nullptr);
            objectTransformTool       = ObjectTransformTool::None;
            objectTransformWindowOpen = false;
        }

        lr::SceneObject *replacement = sceneManager.removeSceneObjects(ids);
        sceneManager.uploadLights();
        pbrPass.setNumLights(sceneManager.numLights());
        physicsWorld.onSceneChanged();
        if (!replacement)
        {
            sceneManager.setEditorMode(lr::EditorMode::View);
            return;
        }

        meshObject = replacement;
        lr::Mesh &mesh = replacement->getComponent<lr::MeshComponent>().mesh();
        vertexManager.rebind(mesh);
        arapTool.rebind(mesh);
        laplaceBeltramiTool.rebind(mesh);
        heatmapPass.setMeshSource(sceneManager.selectedMeshHeatmap(), sceneManager.selectedMeshIndexRange(),
                                  replacement->getComponent<lr::TransformComponent>());
        overlayPointsPass.setPointsSource(sceneManager.selectedMeshPoints(), mesh.uniquePositionCount(),
                                          replacement->getComponent<lr::TransformComponent>());
    });

    // Single combined LMB handler: gizmos get first refusal on a click (so
    // dragging an arrow doesn't simultaneously start a box-select), and
    // selection only sees the event if no gizmo consumed it.
    viewer.input().onMouseButton([&](int button, int action, bool shift, bool ctrl, bool alt) {
        if (button != GLFW_MOUSE_BUTTON_LEFT || ImGui::GetIO().WantCaptureMouse)
        {
            return;
        }

        const bool wasInteracting = gizmoManager.isInteracting();
        gizmoManager.mouseButtonCallback(button, action, shift, ctrl, alt);

        if (wasInteracting || gizmoManager.isInteracting() || sceneManager.editorMode() != lr::EditorMode::Edit)
        {
            return;
        }

        selectionManager.mouseButtonCallback(button, action, shift, ctrl, alt);
    });

    viewer.input().onKeyPress([&](int key, int action, bool shift, bool ctrl, bool alt) {
        if (key != GLFW_KEY_TAB || action != GLFW_PRESS)
        {
            return;
        }
        if (ImGui::GetIO().WantCaptureKeyboard)
        {
            return;
        }

        const bool nowEditing = sceneManager.editorMode() != lr::EditorMode::Edit;
        if (nowEditing)
        {
            // Entering Edit mode requires a mesh object to actually be selected — otherwise
            // there's nothing for the vertex-picking overlay/gizmos to operate on.
            const auto selected = scene.selectedObject();
            if (!selected || !lr::SceneManager::isEditable(scene.getSceneObject(selected.value())))
            {
                return;
            }
        }
        sceneManager.setEditorMode(nowEditing ? lr::EditorMode::Edit : lr::EditorMode::View);

        // ARAP mode deliberately outlives Edit mode now — a solved handle set should stay
        // draggable in View mode. Anchor/handle (re)classification still requires a selection,
        // which only Edit mode's box-select can produce, so the solve can't go stale outside Edit.
    });

    viewer.input().onKeyPress([&](int key, int action, bool shift, bool ctrl, bool alt) {
        if (key != GLFW_KEY_Z || action != GLFW_PRESS || !ctrl)
        {
            return;
        }
        if (ImGui::GetIO().WantCaptureKeyboard)
        {
            return;
        }

        commandManager.undo();
    });

    viewer.input().onKeyPress([&](int key, int action, bool shift, bool ctrl, bool alt) {
        if (key != GLFW_KEY_DELETE || action != GLFW_PRESS || ImGui::GetIO().WantCaptureKeyboard)
        {
            return;
        }

        const auto selected = scene.selectedObject();
        if (selected && scene.canDestroySceneObject(*selected))
        {
            scene.destroySceneObject(*selected);
        }
    });

    viewer.input().onKeyPress([&](int key, int action, bool shift, bool ctrl, bool alt) {
        if (key != GLFW_KEY_A || action != GLFW_PRESS)
        {
            return;
        }
        if (ImGui::GetIO().WantCaptureKeyboard)
        {
            return;
        }

        arapTool.setModeActive(!arapTool.isModeActive());
    });

    // -------------------------------------------------------------------------
    // Per-frame callbacks
    // -------------------------------------------------------------------------

    std::optional<fs::path> environmentHdriPath;
    std::string             environmentLoadError;
    bool                    environmentDirty = false;
    std::string             sceneImportError;

    viewer.onGui([&]() {
        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        const ImVec2 panelSize(viewport->WorkSize.x * 0.24f, viewport->WorkSize.y * 0.32f);
        const ImVec2 topLeft(viewport->WorkPos.x, viewport->WorkPos.y);
        const ImVec2 topRight(viewport->WorkPos.x + viewport->WorkSize.x - panelSize.x,
                              viewport->WorkPos.y);
        const ImVec2 bottomLeft(viewport->WorkPos.x,
                                viewport->WorkPos.y + viewport->WorkSize.y - panelSize.y);
        const ImVec2 bottomRight(viewport->WorkPos.x + viewport->WorkSize.x - panelSize.x,
                                 viewport->WorkPos.y + viewport->WorkSize.y - panelSize.y);

        laplaceBeltramiTool.onGui();

        ImGui::SetNextWindowPos(topLeft, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(panelSize, ImGuiCond_FirstUseEver);
        ImGui::Begin("Physics");
        physicsWorld.onGUI();
        ImGui::End();

        ImGui::SetNextWindowPos(bottomLeft, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(panelSize, ImGuiCond_FirstUseEver);
        ImGui::Begin("Environment");

        if (ImGui::CollapsingHeader("Environment", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Indent();

            ImGui::TextUnformatted("HDRI");
            ImGui::SameLine();
            if (environmentHdriPath)
            {
                ImGui::TextWrapped("%s", environmentHdriPath->filename().string().c_str());
            } else
            {
                ImGui::TextDisabled("None (black environment)");
            }

            if (ImGui::Button("Load HDRI..."))
            {
                IGFD::FileDialogConfig dialogConfig;
                dialogConfig.path  = environmentHdriPath ? environmentHdriPath->parent_path().string() : ".";
                dialogConfig.flags = ImGuiFileDialogFlags_Modal | ImGuiFileDialogFlags_ReadOnlyFileNameField |
                                     ImGuiFileDialogFlags_CaseInsensitiveExtentionFiltering |
                                     ImGuiFileDialogFlags_ShowDevicesButton;
                ImGuiFileDialog::Instance()->OpenDialog("ChooseEnvironmentHdri", "Select HDRI", ".hdr", dialogConfig);
            }
            ImGui::SameLine();

            ImGui::BeginDisabled(!environmentHdriPath.has_value());
            if (ImGui::Button("Clear"))
            {
                environmentHdriPath.reset();
                environmentLoadError.clear();
                environmentDirty = true;
            }
            ImGui::EndDisabled();

            ImGui::Spacing();
            ImGui::TextDisabled("Supported format: Radiance HDR (.hdr)");
            if (!environmentLoadError.empty())
            {
                ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Load failed: %s", environmentLoadError.c_str());
            }

            ImGui::Unindent();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseEnvironmentHdri", ImGuiWindowFlags_NoCollapse,
                                                 ImVec2(640.0f, 360.0f)))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                fs::path selectedPath(ImGuiFileDialog::Instance()->GetFilePathName());
                if (environmentHdriPath != selectedPath)
                {
                    environmentHdriPath = std::move(selectedPath);
                    environmentLoadError.clear();
                    environmentDirty = true;
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        ImGui::End();

        ImGui::Begin("HBAO");
        {
            lr::AmbientOcclusionPass::Config &aoConfig = aoPass.config();
            bool                              aoDirty  = false;
            aoDirty |= ImGui::SliderFloat("Sphere Radius", &aoConfig.sphereRadius, 0.0005f, 0.2f, "%.4f",
                                          ImGuiSliderFlags_Logarithmic);
            aoDirty |= ImGui::SliderInt("Num Steps", &aoConfig.numSteps, 1, 128);
            aoDirty |= ImGui::SliderInt("Num Directions", &aoConfig.numDirs, 1, 128);
            aoDirty |= ImGui::SliderFloat("Tan Angle Bias", &aoConfig.tanAngleBias, 0.0f, 1.0f);
            aoDirty |= ImGui::SliderFloat("AO Scalar", &aoConfig.aoScalar, 0.0f, 5.0f);
            if (aoDirty)
            {
                aoPass.updateParams(viewer.resources());
            }
        }
        ImGui::End();

        ImGui::SetNextWindowPos(topRight, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(panelSize, ImGuiCond_FirstUseEver);
        ImGui::Begin("Scene Hierarchy");
        if (ImGui::Button("Import..."))
        {
            IGFD::FileDialogConfig dialogConfig;
            dialogConfig.path  = ".";
            dialogConfig.flags = ImGuiFileDialogFlags_Modal | ImGuiFileDialogFlags_ReadOnlyFileNameField |
                                 ImGuiFileDialogFlags_CaseInsensitiveExtentionFiltering |
                                 ImGuiFileDialogFlags_ShowDevicesButton;
            ImGuiFileDialog::Instance()->OpenDialog("ImportScene", "Import Mesh or Animated Mesh",
                                                     ".obj,.gltf,.glb", dialogConfig);
        }
        if (!sceneImportError.empty())
        {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Import failed: %s", sceneImportError.c_str());
        }
        ImGui::Separator();
        scene.onHierarchyGUI();
        ImGui::End();

        if (ImGuiFileDialog::Instance()->Display("ImportScene", ImGuiWindowFlags_NoCollapse,
                                                  ImVec2(640.0f, 360.0f)))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                try
                {
                    const fs::path selectedPath(ImGuiFileDialog::Instance()->GetFilePathName());
                    viewer.context().waitIdle();
                    sceneManager.load(selectedPath, sceneLoadConfig);
                    sceneManager.rebuildGeometry();
                    geometryPass.setSceneGeometry(sceneManager.meshPositions(), sceneManager.indexBuffer(),
                                                  sceneManager.meshTransforms(), sceneManager.geometryObjects(),
                                                  sceneManager.skinUploadResult().drawInfos);
                    if (lr::SceneObject *selectedMesh = sceneManager.editedMeshObject())
                    {
                        heatmapPass.setMeshSource(sceneManager.selectedMeshHeatmap(),
                                                  sceneManager.selectedMeshIndexRange(),
                                                  selectedMesh->getComponent<lr::TransformComponent>());
                    }
                    physicsWorld.onSceneChanged();
                    viewer.frameGraph().compile();
                    sceneImportError.clear();
                }
                catch (const std::exception &e)
                {
                    sceneImportError = e.what();
                    spdlog::error("Failed to import scene: {}", e.what());
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        ImGui::SetNextWindowPos(bottomRight, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(panelSize, ImGuiCond_FirstUseEver);
        ImGui::Begin("Inspector");
        scene.onInspectorGUI();
        ImGui::End();

        if (objectTransformWindowOpen && scene.selectedObject())
        {
            ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                           viewport->WorkPos.y + 20.0f),
                                    ImGuiCond_Appearing, ImVec2(0.5f, 0.0f));
            ImGui::SetNextWindowSize(ImVec2(330.0f, 0.0f), ImGuiCond_Appearing);
            if (ImGui::Begin("Transform Gizmo", &objectTransformWindowOpen, ImGuiWindowFlags_AlwaysAutoResize))
            {
                const bool canTranslate = sceneObjectHandler.target() != nullptr;
                ImGui::BeginDisabled(!canTranslate);
                if (ImGui::RadioButton("Translate", objectTransformTool == ObjectTransformTool::Translate))
                {
                    objectTransformTool = ObjectTransformTool::Translate;
                }
                ImGui::EndDisabled();

                ImGui::SameLine();
                ImGui::BeginDisabled();
                ImGui::RadioButton("Rotate", false);
                ImGui::SameLine();
                ImGui::RadioButton("Scale", false);
                ImGui::EndDisabled();
            }
            ImGui::End();
        }
    });

    // Application-level rendering policy: the C++ demo owns the IBL shaders, resource names,
    // and rebuild timing. A future Python entry point can express the same composition using the
    // frame-graph and resource-registry bindings without Scene knowing what an environment is.
    viewer.onLateUpdate([&](float, VkExtent2D) {
        if (!environmentDirty)
        {
            return;
        }

        try
        {
            auto reloadConfig     = iblConfig;
            reloadConfig.hdriPath = environmentHdriPath.value_or(fs::path{});
            lr::IBLPass iblPass(std::move(reloadConfig));

            // The existing cubemaps are sampled by in-flight frames and overwritten in place.
            viewer.context().waitIdle();
            iblPass.replaceHdriResource(viewer.resources());

            lr::FrameGraph iblGraph(viewer.context(), viewer.resources());
            iblPass.build(iblGraph);
            iblGraph.executeAndWait({
                {"ibl_irradiance", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                {"ibl_prefiltered", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            });

            environmentLoadError.clear();
        } catch (const std::exception &e)
        {
            environmentLoadError = e.what();
            spdlog::error("Failed to update environment: {}", e.what());
        }

        environmentDirty = false;
    });

    lr::SphericalCameraController cameraController(*camera, viewer.input());
    viewer.onUpdate([&](float dt, VkExtent2D) {
        physicsWorld.update(dt);
    });

    viewer.onUpdate([&cameraController](float dt, VkExtent2D extent) {
        cameraController.update(dt);
    });

    viewer.onUpdate([&](float dt, VkExtent2D extent) {
        gizmoManager.updateCallback(dt, extent, viewer.hasRenderedAtLeastOneFrame(), gizmoReadback, viewer.resources());
    });

    // selectionManager's per-frame mouse/drag handling is driven by SceneManager::registerCallbacks()
    // below, since SceneManager now owns the SelectionManager instance.

    // Drives the ARAP anchor/handle popup + Solve button, and keeps the translate gizmos
    // positioned at whichever index set the currently-wired target drives (the selection by
    // default, or the handle set once an ARAP precompute has succeeded) — shown only while that
    // set is non-empty, and pushes the result to the overlay pass.
    viewer.onUpdate([&](float dt, VkExtent2D extent) {
        const float aspect =
            (extent.height == 0) ? 1.0f : static_cast<float>(extent.width) / static_cast<float>(extent.height);
        const glm::mat4 viewProj = camera->getComponent<lr::Camera>().viewProjectionMatrix(aspect);

        // Average in local space, then transform once — valid since centroid-of-transformed-points
        // equals transform-of-centroid for any affine map. The result is a world-space point since
        // the gizmo/popup aren't part of the mesh, so GeometryPass's model matrix never applies to
        // them — they need to track where the vertices actually render, not their local positions.
        auto worldCentroidOf = [&](const std::unordered_set<uint32_t> &idxs) {
            glm::vec3 localCentroid(0.0f);
            for (uint32_t idx : idxs)
            {
                localCentroid += vertexManager.getPositions()[idx];
            }
            localCentroid /= static_cast<float>(idxs.size());
            return glm::vec3(meshObject->getComponent<lr::TransformComponent>().worldMatrix() *
                             glm::vec4(localCentroid, 1.0f));
        };

        const auto &selected = selectionManager.getSelectedIndices();
        arapTool.onGui(viewProj, extent, selected.empty() ? glm::vec3(0.0f) : worldCentroidOf(selected));

        const bool objectTranslateActive = objectTransformTool == ObjectTransformTool::Translate &&
                                           sceneManager.editorMode() == lr::EditorMode::View &&
                                           sceneObjectHandler.target() != nullptr;
        if (objectTranslateActive)
        {
            if (&arrowX->dragHandler() != &sceneObjectHandler)
            {
                objectTranslateReturnHandler = &arrowX->dragHandler();
            }
            for (lr::TranslateDragHandlerGizmo *gizmo : dragHandlerGizmos)
            {
                gizmo->setDragHandler(sceneObjectHandler);
            }
        } else if (&arrowX->dragHandler() == &sceneObjectHandler)
        {
            for (lr::TranslateDragHandlerGizmo *gizmo : dragHandlerGizmos)
            {
                gizmo->setDragHandler(*objectTranslateReturnHandler);
            }
        }

        // All 4 gizmos always share the same handler (ArapTool swaps them together), so any one
        // of them tells us which is currently active.
        const lr::TranslateDragHandler &activeHandler = arrowX->dragHandler();
        const auto &driven = objectTranslateActive
                                 ? selectionManager.getSelectedIndices()
                                 : static_cast<const lr::VertexDragHandler &>(activeHandler).indices();
        // While ARAP mode is active but no precompute has succeeded yet, the default drag gizmo
        // would otherwise appear over the very selection the anchor/handle popup is asking about —
        // suppress it until Solve actually swaps the handler.
        const bool suppressedByArapMode = arapTool.isModeActive() && (&activeHandler == &defaultHandler);
        // Deliberately not gated on Edit mode: an ARAP-solved handle set should stay draggable in
        // View mode too (see the Tab handler above — ARAP mode now outlives Edit). This still can't
        // leak the plain translate gizmo into View mode, since the default handler's indices are the
        // current selection, which SceneManager clears on leaving Edit — so driven.empty() already
        // covers that case on its own.
        if ((!objectTranslateActive && driven.empty()) || suppressedByArapMode)
        {
            for (int id : translateGizmoIds)
            {
                gizmoManager.hideGizmo(id);
            }
        } else
        {
            const glm::vec3 centroid = objectTranslateActive
                                           ? glm::vec3(sceneObjectHandler.target()->worldMatrix()[3])
                                           : worldCentroidOf(driven);

            // Keep the gizmo a constant size on screen (~1/9 screen height) regardless of camera distance.
            const glm::vec3 camPos = camera->getComponent<lr::TransformComponent>().transform().position();
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
                gizmo.setScale(i < 3 ? glm::vec3(rad, len, rad) : glm::vec3(rad * 0.45f, rad * 0.45f, rad * 0.45f));
            }
        }

        overlayGeometryPass.setInstances(gizmoManager.getVisibleGizmoInstances());
        overlayLinesPass.setLines(lr::buildColliderOverlayLines(scene));
    });

    // Registers SceneManager's own onUpdate (aspect tracking) and onLateUpdate (flushDirty —
    // runs after every onUpdate above, so it sees the results of this frame's camera
    // controller / gizmo / GUI edits and does at most one GPU re-upload per dirtied resource
    // rather than one per individual mutation) callbacks — see SceneManager::registerCallbacks.
    sceneManager.registerCallbacks(viewer);

    viewer.addImguiPass();
    viewer.run();
    return 0;
} catch (const std::exception &e)
{
    spdlog::error("Fatal: {}", e.what());
    throw;
    return 1;
}
