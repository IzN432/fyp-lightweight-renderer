#include "app/EditorSession.hpp"
#include "core/app/Viewer.hpp"
#include "core/Paths.hpp"
#include "core/loaders/Material.hpp"
#include "core/loaders/MaterialStore.hpp"
#include "core/passes/final/FinalPass.hpp"
#include "core/passes/geometry/GeometryPass.hpp"
#include "core/passes/heatmap/HeatmapPass.hpp"
#include "core/passes/ibl/IblPass.hpp"
#include "core/passes/pbr/PbrPass.hpp"
#include "core/passes/transparent/TransparentPass.hpp"
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
#include "core/scene/EngineConventions.hpp"
#include "core/scene/SceneManager.hpp"
#include "core/scene/Scene.hpp"

#include "features/rigid_body/ColliderComponent.hpp"
#include "features/rigid_body/PhysicsWorld.hpp"
#include "features/rigid_body/RigidBodyComponent.hpp"

#include <ImGuiFileDialog.h>
#include <imgui.h>
#include <glm/vec4.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
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

    // The editor pipeline currently needs one mesh while its GPU buffers and mesh-editing tools are
    // constructed. This bootstrap asset is retired before the frame loop, leaving only Main Camera
    // in the initial scene hierarchy.
    const fs::path meshPath = lr::paths::assetDir / "samples/models/bird_orange.glb";
    lr::SceneObject &bootstrapRoot = sceneManager.load(meshPath, sceneLoadConfig);

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

    // glTF BLEND materials: GeometryPass discards them, and this pass shades them back to front into the
    // "transparent" layer FinalPass composites over the lit scene.
    lr::MaterialStore  &materialStore = sceneManager.materialStore();
    lr::TransparentPass transparentPass({
        .geometry                = sceneManager.gpu().geometryPassConfig(),
        .lightBufferResourceName = sceneManager.lightBufferName(),
        .numLights               = sceneManager.numLights(),
        .pfMips                  = 8,
        .eyePosition =
            [&sceneManager] {
                const lr::Camera &camera = sceneManager.gpu().camera()->getComponent<lr::Camera>();
                return glm::vec3(glm::inverse(camera.viewMatrix())[3]);
            },
        .isBlendMaterial =
            [&materialStore](lr::MaterialHandle handle) {
                const auto &parameters = materialStore.get(handle).parameters;
                const auto  found      = parameters.find(lr::conventions::alphaBlend);
                const auto *flag       = found == parameters.end()
                                             ? nullptr
                                             : std::get_if<lr::MaterialParam::NormalizedFloat>(&found->second);
                return flag && flag->value > 0.5f;
            },
    });
    const auto          setTransparentGeometry = [&](const lr::SceneGpu &gpu) {
        transparentPass.setSceneGeometry(gpu.meshPositions(), gpu.indexBuffer(), gpu.meshTransforms(),
                                         gpu.geometryObjects(), gpu.skinUploadResult().drawInfos, gpu.geometryMeshes());
    };
    setTransparentGeometry(sceneManager.gpu());
    transparentPass.build(viewer.frameGraph(), lr::conventions::geometryMeshLayout());

    // Whenever SceneGpu re-packs geometry (an import, lights added or removed) or re-uploads the lights,
    // keep the passes' draw lists and light count in step.
    sceneManager.gpu().onGeometryRebuilt([&](const lr::SceneGpu &gpu) {
        geometryPass.setSceneGeometry(gpu.meshPositions(), gpu.indexBuffer(), gpu.meshTransforms(),
                                      gpu.geometryObjects(), gpu.skinUploadResult().drawInfos);
        setTransparentGeometry(gpu);
    });
    sceneManager.gpu().onLightsUploaded([&](uint32_t numLights) {
        pbrPass.setNumLights(numLights);
        transparentPass.setNumLights(numLights);
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

    // Editor interaction is composed behind one facade. A future Engine can own this alongside
    // the render pipeline without reintroducing editor orchestration into main.cpp.
    lr::EditorSession editor(viewer, sceneManager, *camera, geometryPass, transparentPass,
                             heatmapPass, overlayPointsPass, overlayLinesPass);
    scene.registerObjectsDestroyedCallback([&](std::span<const lr::SceneObjectId>) {
        physicsWorld.onSceneChanged();
    });

    // -------------------------------------------------------------------------
    // Per-frame callbacks
    // -------------------------------------------------------------------------

    glm::vec3               environmentBackgroundColor(0.0f);
    std::string             environmentLoadError;
    bool                    environmentDirty = false;
    std::string             sceneImportError;
    std::optional<fs::path> sceneDocumentPath;
    std::string             scenePersistenceError;

    viewer.onGui([&]() {
        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                if (ImGui::MenuItem("Load Scene..."))
                {
                    IGFD::FileDialogConfig dialogConfig;
                    dialogConfig.path = sceneDocumentPath ? sceneDocumentPath->parent_path().string() : ".";
                    dialogConfig.flags = ImGuiFileDialogFlags_Modal |
                                         ImGuiFileDialogFlags_CaseInsensitiveExtentionFiltering |
                                         ImGuiFileDialogFlags_ShowDevicesButton;
                    ImGuiFileDialog::Instance()->OpenDialog("LoadNativeScene", "Load Scene", ".lrscene",
                                                             dialogConfig);
                }

                if (ImGui::MenuItem("Save", nullptr, false, sceneDocumentPath.has_value()))
                {
                    try
                    {
                        sceneManager.save(*sceneDocumentPath);
                        scenePersistenceError.clear();
                    }
                    catch (const std::exception &e)
                    {
                        scenePersistenceError = e.what();
                        spdlog::error("Failed to save scene: {}", e.what());
                    }
                }

                if (ImGui::MenuItem("Save As..."))
                {
                    IGFD::FileDialogConfig dialogConfig;
                    dialogConfig.path = sceneDocumentPath ? sceneDocumentPath->parent_path().string() : ".";
                    dialogConfig.fileName = sceneDocumentPath ? sceneDocumentPath->filename().string()
                                                                : "scene.lrscene";
                    dialogConfig.flags = ImGuiFileDialogFlags_Modal | ImGuiFileDialogFlags_ConfirmOverwrite |
                                         ImGuiFileDialogFlags_CaseInsensitiveExtentionFiltering |
                                         ImGuiFileDialogFlags_ShowDevicesButton;
                    ImGuiFileDialog::Instance()->OpenDialog("SaveNativeScene", "Save Scene As", ".lrscene",
                                                             dialogConfig);
                }
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }

        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        const ImVec2 panelSize(viewport->WorkSize.x * 0.24f, viewport->WorkSize.y * 0.32f);
        const ImVec2 topLeft(viewport->WorkPos.x, viewport->WorkPos.y);
        const ImVec2 topRight(viewport->WorkPos.x + viewport->WorkSize.x - panelSize.x,
                              viewport->WorkPos.y);
        const ImVec2 bottomLeft(viewport->WorkPos.x,
                                viewport->WorkPos.y + viewport->WorkSize.y - panelSize.y);
        const ImVec2 bottomRight(viewport->WorkPos.x + viewport->WorkSize.x - panelSize.x,
                                 viewport->WorkPos.y + viewport->WorkSize.y - panelSize.y);

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
            if (scene.hdriPath())
            {
                ImGui::TextWrapped("%s", scene.hdriPath()->filename().string().c_str());
            } else
            {
                ImGui::TextDisabled("None (background color)");
            }

            if (ImGui::Button("Load HDRI..."))
            {
                IGFD::FileDialogConfig dialogConfig;
                dialogConfig.path  = scene.hdriPath() ? scene.hdriPath()->parent_path().string() : ".";
                dialogConfig.flags = ImGuiFileDialogFlags_Modal | ImGuiFileDialogFlags_ReadOnlyFileNameField |
                                     ImGuiFileDialogFlags_CaseInsensitiveExtentionFiltering |
                                     ImGuiFileDialogFlags_ShowDevicesButton;
                ImGuiFileDialog::Instance()->OpenDialog("ChooseEnvironmentHdri", "Select HDRI", ".hdr", dialogConfig);
            }
            ImGui::SameLine();

            ImGui::BeginDisabled(!scene.hdriPath().has_value());
            if (ImGui::Button("Clear"))
            {
                scene.setHdriPath(std::nullopt);
                environmentLoadError.clear();
                environmentDirty = true;
            }
            ImGui::EndDisabled();

            ImGui::Spacing();
            ImGui::BeginDisabled(scene.hdriPath().has_value());
            if (ImGui::ColorEdit3("Background color", &environmentBackgroundColor.x))
            {
                finalPass.setBackground(glm::vec4(environmentBackgroundColor, 1.0f), false);
            }
            ImGui::EndDisabled();
            if (scene.hdriPath() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
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

        if (ImGui::CollapsingHeader("HBAO"))
        {
            ImGui::Indent();
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
            ImGui::Unindent();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseEnvironmentHdri", ImGuiWindowFlags_NoCollapse,
                                                 ImVec2(640.0f, 360.0f)))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                fs::path selectedPath(ImGuiFileDialog::Instance()->GetFilePathName());
                if (scene.hdriPath() != selectedPath)
                {
                    scene.setHdriPath(std::move(selectedPath));
                    environmentLoadError.clear();
                    environmentDirty = true;
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        ImGui::End();

        ImGui::SetNextWindowPos(topRight, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(panelSize.x, viewport->WorkSize.y * 0.64f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Scene");
        const float hierarchyHeight = ImGui::GetContentRegionAvail().y * 0.55f;
        ImGui::BeginChild("SceneHierarchy", ImVec2(0.0f, hierarchyHeight),
                          ImGuiChildFlags_Borders | ImGuiChildFlags_ResizeY);
        ImGui::SeparatorText("Hierarchy");
        if (!scenePersistenceError.empty())
        {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Scene file error: %s",
                               scenePersistenceError.c_str());
        }
        ImGui::Separator();
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
        ImGui::EndChild();

        ImGui::BeginChild("SceneInspector", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders);
        ImGui::SeparatorText("Inspector");
        scene.onInspectorGUI(editor.context());
        ImGui::EndChild();
        ImGui::End();

        if (ImGuiFileDialog::Instance()->Display("LoadNativeScene", ImGuiWindowFlags_NoCollapse,
                                                  ImVec2(640.0f, 360.0f)))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                try
                {
                    fs::path selectedPath(ImGuiFileDialog::Instance()->GetFilePathName());
                    const fs::path scenePath = selectedPath;
                    viewer.context().waitIdle();
                    sceneManager.loadScene(scenePath);
                    environmentDirty = true;
                    sceneManager.rebuildGeometry();
                    editor.onSceneContentChanged();
                    physicsWorld.onSceneChanged();
                    viewer.frameGraph().compile();
                    sceneDocumentPath = scenePath;
                    scenePersistenceError.clear();
                }
                catch (const std::exception &e)
                {
                    scenePersistenceError = e.what();
                    spdlog::error("Failed to load scene: {}", e.what());
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("SaveNativeScene", ImGuiWindowFlags_NoCollapse,
                                                  ImVec2(640.0f, 360.0f)))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                try
                {
                    fs::path selectedPath(ImGuiFileDialog::Instance()->GetFilePathName());
                    if (selectedPath.extension() != ".lrscene") selectedPath += ".lrscene";
                    sceneManager.save(selectedPath);
                    sceneDocumentPath = std::move(selectedPath);
                    scenePersistenceError.clear();
                }
                catch (const std::exception &e)
                {
                    scenePersistenceError = e.what();
                    spdlog::error("Failed to save scene: {}", e.what());
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

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
                    editor.onSceneContentChanged();
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
        ImGui::Begin("Features");
        editor.drawFeaturePanel();
        ImGui::End();
        editor.drawTransformWindow();
    });

    // Application-level rendering policy: the C++ demo owns the IBL shaders, resource names,
    // and rebuild timing. Scene only retains the authored HDRI path for persistence.
    viewer.onLateUpdate([&](float, VkExtent2D) {
        if (!environmentDirty)
        {
            return;
        }

        try
        {
            auto reloadConfig     = iblConfig;
            reloadConfig.hdriPath = scene.hdriPath().value_or(fs::path{});
            reloadConfig.hdriData = scene.hdriData();
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
                                    scene.hdriPath().has_value());

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
        cameraController.update(dt, editor.allowsViewportNavigation());
    });

    // Registers SceneManager's own onUpdate (aspect tracking) and onLateUpdate (flushDirty —
    // runs after every onUpdate above, so it sees the results of this frame's camera
    // controller / gizmo / GUI edits and does at most one GPU re-upload per dirtied resource
    // rather than one per individual mutation) callbacks — see SceneManager::registerCallbacks.
    sceneManager.registerCallbacks(viewer);

    // Start with no authored scene objects. Retirement keeps the bootstrap allocations alive long
    // enough for dormant editor references to remain safe until loadScene() clears the asset stores
    // and rebinds every mesh-editing tool to the loaded scene.
    scene.destroySceneObject(bootstrapRoot.id());

    viewer.addImguiPass();
    viewer.run();
    return 0;
} catch (const std::exception &e)
{
    spdlog::error("Fatal: {}", e.what());
    throw;
    return 1;
}
