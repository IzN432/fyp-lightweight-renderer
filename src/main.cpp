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

#include "core/scene/AreaLightVisual.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/editor/camera/SphericalCameraController.hpp"
#include "core/editor/gizmo/RotateGizmo.hpp"
#include "core/editor/gizmo/ScaleGizmo.hpp"
#include "core/editor/gizmo/TranslateGizmo.hpp"
#include "core/editor/SceneObjectDragHandler.hpp"
#include "core/editor/SceneObjectRotationHandler.hpp"
#include "core/editor/SceneObjectScaleHandler.hpp"
#include "core/editor/SceneObjectTransformController.hpp"
#include "core/editor/selection/BoxSelectionTool.hpp"
#include "core/editor/selection/SelectionManager.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/editor/DefaultVertexDragHandler.hpp"
#include "core/scene/EngineConventions.hpp"
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

    // Attribute/texture/parameter names shared by the loaders, GeometryPass and the Python module —
    // see EngineConventions.hpp.
    const lr::GltfLoaderConfig  config          = lr::conventions::gltfLoaderConfig();
    const lr::SceneLoaderConfig sceneLoadConfig = lr::conventions::sceneLoaderConfig();

    lr::SceneManager sceneManager(viewer.resources(), lr::conventions::materialCapacity,
                                  lr::conventions::defaultMaterial);
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
    const fs::path meshPath = lr::paths::assetDir / "samples/models/bird_orange.glb";

    sceneManager.load(meshPath, sceneLoadConfig);

    // LIGHT VISUALS — every light, not just ones that start out as AreaLight, gets a quad mesh owned by
    // SceneGpu (not a component on the light, so it isn't selectable or editable). The quad still draws
    // through the same GeometryPass (see AreaLightVisual.hpp for why the visual needs to be real
    // geometry rather than an overlay); its material lives in a MaterialStore slot acquired up front,
    // so switching a light's type at runtime (see Light::onGUIImpl) just rewrites that slot in place —
    // see SceneGpu::updateLightVisuals.
    const lr::AreaLightVisualConfig areaLightVisualConfig = lr::conventions::areaLightVisualConfig();

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

    }

    lr::PhysicsWorld physicsWorld(scene);

    // -------------------------------------------------------------------------
    // Resource uploads
    // -------------------------------------------------------------------------

    // This matches the expected layout in geometry.frag
    const lr::GpuMaterialLayout gpuMaterialLayout = lr::conventions::materialLayout();

    // Builds light visuals, uploads the initial lights/mesh/material/camera buffers, and wires the
    // change listeners that keep the camera UBO and materials SSBO in sync afterward —
    // see SceneManager::initialize().
    sceneManager.initialize(areaLightVisualConfig, gpuMaterialLayout, lr::conventions::geometryVertexAttributes(),
                            viewer.input());

    // -------------------------------------------------------------------------
    // Frame graph passes
    // -------------------------------------------------------------------------

    const VkFormat swapchainFormat = viewer.frameGraph().resources().getImage("swapchain")->format;

    // Draws everything SceneGpu uploaded (see SceneGpu::geometryPassConfig) with the engine's vertex
    // layout (conventions::geometryMeshLayout).
    lr::GeometryPass geometryPass(sceneManager.gpu().geometryPassConfig());
    geometryPass.build(viewer.frameGraph(), lr::conventions::geometryMeshLayout());

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
    heatmapMeshLayout.mapUniqueVertex("heatmapColors", 0, 1, VK_FORMAT_R32G32B32_SFLOAT);

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

    // Whenever SceneGpu re-packs geometry (an import, lights added or removed) or re-uploads the lights,
    // keep the passes' draw lists and light count in step.
    sceneManager.gpu().onGeometryRebuilt([&](const lr::SceneGpu &gpu) {
        geometryPass.setSceneGeometry(gpu.meshPositions(), gpu.indexBuffer(), gpu.meshTransforms(),
                                      gpu.geometryObjects(), gpu.skinUploadResult().drawInfos);
    });
    sceneManager.gpu().onLightsUploaded([&](uint32_t numLights) {
        pbrPass.setNumLights(numLights);
    });

    // Keep the geometry overlay stage alive even when it has no instances. Besides remaining
    // available for future editor visuals (for example, bones), it owns the per-frame clear of
    // the shared overlay color/depth targets before later overlay passes append to them.
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
        .showEnvironmentBackground = false,
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

    lr::VertexManager vertexManager(meshComponent.mesh());

    lr::CommandManager commandManager;

    // SceneManager owns the SelectionManager (constructed for the initial selected mesh, see
    // SceneManager::selectionManager()) since it needs to wire selection-highlight changes straight
    // to the GPU color buffer; this is just a local alias to keep the call sites below unchanged.
    lr::SelectionManager &selectionManager = sceneManager.selectionManager();
    selectionManager.setSelectTool(std::make_unique<lr::BoxSelectionTool>(viewer.input(), *camera, selectionManager));

    // What the translation gizmo drives by default. ArapTool swaps this for its solve handler
    // after a successful precompute.
    lr::DefaultVertexDragHandler defaultHandler(vertexManager, selectionManager, commandManager);
    lr::SceneObjectDragHandler   sceneObjectHandler(commandManager);
    lr::SceneObjectRotationHandler sceneObjectRotationHandler(commandManager);
    lr::SceneObjectScaleHandler sceneObjectScaleHandler(commandManager);
    lr::TranslateGizmo       translateGizmo(defaultHandler);
    lr::RotateGizmo          rotateGizmo(sceneObjectRotationHandler);
    lr::ScaleGizmo           scaleGizmo(sceneObjectScaleHandler);
    lr::TranslateDragHandler *objectTranslateReturnHandler = &defaultHandler;

    lr::SceneObjectTransformController transformController(
        sceneObjectHandler, sceneObjectRotationHandler, sceneObjectScaleHandler);
    lr::EditorContext editorContext{transformController, commandManager};
    bool                objectTransformWindowOpen = false;

    lr::ArapTool arapTool(selectionManager, vertexManager, commandManager, meshComponent.mesh(), defaultHandler,
                          translateGizmo);

    lr::LaplaceBeltramiTool laplaceBeltramiTool(meshComponent.mesh(), sceneManager);

    // Ties vertex editing (SelectionManager, VertexManager, ArapTool, the vertex-picking points
    // overlay) to whatever's selected in the Scene Hierarchy. ARAP, Laplace-Beltrami analysis,
    // picking, and both selected-mesh GPU overlays are all
    // rebound together so every editing/analysis operation addresses the same object.
    scene.registerSelectionChangedCallback([&](lr::SceneObjectId id) {
        lr::SceneObject &object = scene.getSceneObject(id);
        objectTransformWindowOpen = true;
        transformController.setSelectedTarget(object.hasComponent<lr::TransformComponent>() ? &object : nullptr);
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
        if (transformController.target() &&
            std::ranges::find(ids, transformController.target()->id()) != ids.end())
        {
            objectTransformWindowOpen = false;
        }
        transformController.onObjectsDestroyed(ids);

        lr::SceneObject *replacement = sceneManager.removeSceneObjects(ids);
        sceneManager.uploadLights();
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

    // ImGuizmo gets first refusal so manipulating a handle does not also start a box-select.
    viewer.input().onMouseButton([&](int button, int action, bool shift, bool ctrl, bool alt) {
        if (button != GLFW_MOUSE_BUTTON_LEFT || ImGui::GetIO().WantCaptureMouse)
        {
            return;
        }

        if (translateGizmo.capturesMouse() || rotateGizmo.capturesMouse() || scaleGizmo.capturesMouse() ||
            sceneManager.editorMode() != lr::EditorMode::Edit)
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
    glm::vec3               environmentBackgroundColor(0.0f);
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
        ImGui::Begin("Settings");

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
                ImGui::TextDisabled("None (background color)");
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
            ImGui::BeginDisabled(environmentHdriPath.has_value());
            if (ImGui::ColorEdit3("Background color", &environmentBackgroundColor.x))
            {
                finalPass.setBackground(glm::vec4(environmentBackgroundColor, 1.0f), false);
            }
            ImGui::EndDisabled();
            if (environmentHdriPath && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            {
                ImGui::SetTooltip("Clear the HDRI to use the background color");
            }

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
        ImGui::SameLine();
        ImGui::BeginDisabled(sceneManager.numLights() >= sceneManager.gpu().maxLights());
        if (ImGui::Button("Add Light"))
        {
            lr::SceneObject &lightObject = scene.createSceneObject();
            lightObject.name             = "Light";
            lightObject.addComponent<lr::TransformComponent>();
            lightObject.addComponent<lr::Light>(lr::PointLight{});
        }
        ImGui::EndDisabled();
        if (sceneManager.numLights() >= sceneManager.gpu().maxLights() &&
            ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        {
            ImGui::SetTooltip("The scene already contains the maximum of %u lights",
                              sceneManager.gpu().maxLights());
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
                    sceneManager.rebuildGeometry(); // refreshes geometryPass via onGeometryRebuilt
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
        scene.onInspectorGUI(editorContext);
        ImGui::End();

        if (objectTransformWindowOpen && scene.selectedObject())
        {
            ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                           viewport->WorkPos.y + 20.0f),
                                    ImGuiCond_Appearing, ImVec2(0.5f, 0.0f));
            ImGui::SetNextWindowSize(ImVec2(330.0f, 0.0f), ImGuiCond_Appearing);
            if (ImGui::Begin("Transform Gizmo", &objectTransformWindowOpen, ImGuiWindowFlags_AlwaysAutoResize))
            {
                const bool canTransform = transformController.target() != nullptr;
                ImGui::BeginDisabled(!canTransform || transformController.hasTemporaryEdit());
                if (ImGui::RadioButton("Translate", transformController.tool() == lr::TransformTool::Translate))
                {
                    transformController.setTool(lr::TransformTool::Translate);
                }
                ImGui::SameLine();
                if (ImGui::RadioButton("Rotate", transformController.tool() == lr::TransformTool::Rotate))
                {
                    transformController.setTool(lr::TransformTool::Rotate);
                }
                ImGui::SameLine();
                if (ImGui::RadioButton("Scale", transformController.tool() == lr::TransformTool::Scale))
                {
                    transformController.setTool(lr::TransformTool::Scale);
                }
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

            finalPass.setBackground(glm::vec4(environmentBackgroundColor, 1.0f),
                                    environmentHdriPath.has_value());

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

    viewer.onUpdate([&](float dt, VkExtent2D extent) {
        cameraController.update(dt, translateGizmo.capturesMouse() || rotateGizmo.capturesMouse() ||
                                      scaleGizmo.capturesMouse());
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

        const bool objectTranslateActive = transformController.tool() == lr::TransformTool::Translate &&
                                           sceneManager.editorMode() == lr::EditorMode::View &&
                                           sceneObjectHandler.target() != nullptr;
        const bool objectRotateActive = transformController.tool() == lr::TransformTool::Rotate &&
                                        sceneManager.editorMode() == lr::EditorMode::View &&
                                        sceneObjectRotationHandler.target() != nullptr;
        const bool objectScaleActive = transformController.tool() == lr::TransformTool::Scale &&
                                       sceneManager.editorMode() == lr::EditorMode::View &&
                                       sceneObjectScaleHandler.target() != nullptr;
        if (objectTranslateActive)
        {
            if (&translateGizmo.dragHandler() != &sceneObjectHandler)
            {
                objectTranslateReturnHandler = &translateGizmo.dragHandler();
            }
            translateGizmo.setDragHandler(sceneObjectHandler);
        } else if (&translateGizmo.dragHandler() == &sceneObjectHandler)
        {
            translateGizmo.setDragHandler(*objectTranslateReturnHandler);
        }

        const lr::TranslateDragHandler &activeHandler = translateGizmo.dragHandler();
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
        const bool gizmoVisible = (objectTranslateActive || !driven.empty()) && !suppressedByArapMode;
        glm::vec3  centroid(0.0f);
        if (gizmoVisible)
        {
            centroid = objectTranslateActive ? glm::vec3(sceneObjectHandler.target()->worldMatrix()[3])
                                             : worldCentroidOf(driven);
        }

        const lr::Camera &cameraComponent = camera->getComponent<lr::Camera>();
        translateGizmo.draw(cameraComponent.viewMatrix(), cameraComponent.projectionMatrix(aspect),
                            cameraComponent.projectionType == lr::ProjectionType::Orthographic, extent, centroid,
                            gizmoVisible);
        rotateGizmo.draw(cameraComponent.viewMatrix(), cameraComponent.projectionMatrix(aspect),
                         cameraComponent.projectionType == lr::ProjectionType::Orthographic, extent,
                         objectRotateActive
                             ? glm::translate(glm::mat4(1.0f),
                                              glm::vec3(sceneObjectRotationHandler.target()->worldMatrix()[3])) *
                                   glm::mat4_cast(sceneObjectRotationHandler.target()->worldRotation())
                             : glm::mat4(1.0f),
                         objectRotateActive);
        scaleGizmo.draw(cameraComponent.viewMatrix(), cameraComponent.projectionMatrix(aspect),
                        cameraComponent.projectionType == lr::ProjectionType::Orthographic, extent,
                        objectScaleActive ? sceneObjectScaleHandler.target()->worldMatrix() : glm::mat4(1.0f),
                        objectScaleActive);
        std::vector<lr::OverlayLine> overlayLines = lr::buildColliderOverlayLines(scene);
        if (const std::optional<lr::SceneObjectId> selected = scene.selectedObject())
        {
            lr::OverlayLineBuilder selectionGizmo;
            lr::SelectionGizmoContext selectionGizmoContext{
                .lines          = selectionGizmo,
                .cameraPosition = glm::vec3(camera->worldMatrix()[3]),
                .cameraForward  = camera->worldRotation() * glm::vec3(0.0f, 0.0f, -1.0f),
                .orthographic   = cameraComponent.projectionType == lr::ProjectionType::Orthographic,
            };
            scene.getSceneObject(*selected).onSelectGizmo(selectionGizmoContext);
            std::vector<lr::OverlayLine> selectionLines = selectionGizmo.takeLines();
            overlayLines.insert(overlayLines.end(), selectionLines.begin(), selectionLines.end());
        }
        overlayLinesPass.setLines(overlayLines);
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
