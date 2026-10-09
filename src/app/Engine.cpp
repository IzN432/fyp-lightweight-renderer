#include "Engine.hpp"
#include "app/EditorSession.hpp"
#include "app/EditorRenderBridge.hpp"
#include "app/Settings.hpp"
#include "core/app/Viewer.hpp"
#include "core/loaders/Material.hpp"
#include "core/loaders/MaterialStore.hpp"
#include "core/passes/final/FinalPass.hpp"
#include "core/passes/geometry/GeometryPass.hpp"
#include "core/passes/heatmap/HeatmapPass.hpp"
#include "core/passes/ibl/IblPass.hpp"
#include "core/passes/pbr/PbrPass.hpp"
#include "core/passes/shadow/SpotShadowPass.hpp"
#include "core/passes/shadow/CascadedShadowPass.hpp"
#include "core/passes/shadow/AreaShadowPass.hpp"
#include "core/passes/transparent/TransparentPass.hpp"
#include "core/passes/ambientocclusion/AmbientOcclusionPass.hpp"
#include "core/passes/overlaygeometry/OverlayGeometryPass.hpp"
#include "core/passes/objectpicking/ObjectPickingPass.hpp"
#include "core/passes/outline/OutlinePass.hpp"
#include "core/passes/overlaylines/OverlayLinesPass.hpp"
#include "core/passes/overlaypoints/OverlayPointsPass.hpp"

#include "core/scene/AreaLightVisual.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/editor/camera/SphericalCameraController.hpp"
#include "core/scene/EngineConventions.hpp"
#include "core/scene/SceneManager.hpp"
#include "core/scene/Scene.hpp"

#include "features/rigid_body/PhysicsWorld.hpp"

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

namespace lr
{

namespace
{
// HBAO world-space sampling radius the app starts with, before anything is loaded. Previously this
// was derived from the bootstrap asset's bounding box; the HBAO panel's slider owns it from here.
constexpr float kDefaultAoSphereRadius = 0.02f;
} // namespace

void Engine::run()
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

    lr::SceneAssets sceneAssets;
    lr::Scene &scene = sceneAssets.scene;

    // Attribute/texture/parameter names shared by the loaders, GeometryPass and the Python module —
    // see EngineConventions.hpp.
    const lr::GltfLoaderConfig  config          = lr::conventions::gltfLoaderConfig();
    const lr::SceneLoaderConfig sceneLoadConfig = lr::conventions::sceneLoaderConfig();

    lr::SceneManager sceneManager(viewer.resources(), sceneAssets);

    lr::SceneObject *camera = &scene.createSceneObject();
    camera->addComponent<lr::Camera>();
    camera->addComponent<lr::TransformComponent>();
    auto &cameraController = camera->addComponent<lr::SphericalCameraController>();
    camera->name = "Main Camera";
    sceneManager.setDefaultCamera(*camera);
    scene.protectSceneObject(camera->id());

    // LIGHT VISUALS — every light, not just ones that start out as AreaLight, gets a quad mesh owned by
    // SceneGpu (not a component on the light, so it isn't selectable or editable). The quad still draws
    // through the same GeometryPass (see AreaLightVisual.hpp for why the visual needs to be real
    // geometry rather than an overlay); its material lives in a MaterialStore slot acquired up front,
    // so switching a light's type at runtime (see Light::onGUIImpl) just rewrites that slot in place —
    // see SceneGpu::updateLightVisuals.
    const lr::AreaLightVisualConfig areaLightVisualConfig = lr::conventions::areaLightVisualConfig();

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

    // A single-sample integer target dedicated to editor selection. Keeping this separate from the
    // MSAA G-buffer avoids integer resolve semantics and lets transparent geometry remain pickable.
    lr::ObjectPickingPass objectPickingPass(sceneManager.gpu().geometryPassConfig());
    objectPickingPass.uploadResources(viewer.resources());
    objectPickingPass.build(viewer.frameGraph(), lr::conventions::geometryMeshLayout());

    lr::SpotShadowPass spotShadowPass(viewer.resources(), {
        .geometry = sceneManager.gpu().geometryPassConfig(),
        .lightObjects = sceneManager.gpu().lightObjects(),
    });
    spotShadowPass.build(viewer.frameGraph(), lr::conventions::geometryMeshLayout());

    lr::CascadedShadowPass cascadedShadowPass(viewer.resources(), {
        .geometry = sceneManager.gpu().geometryPassConfig(),
        .lightObjects = sceneManager.gpu().lightObjects(),
        .camera = camera,
    });
    cascadedShadowPass.build(viewer.frameGraph(), lr::conventions::geometryMeshLayout());

    lr::AreaShadowPass areaShadowPass(viewer.resources(), {
        .geometry = sceneManager.gpu().geometryPassConfig(),
        .lightObjects = sceneManager.gpu().lightObjects(),
    });
    areaShadowPass.build(viewer.frameGraph(), lr::conventions::geometryMeshLayout());

    // Mesh-independent sources and layout: the scene starts empty, and EditorRenderBridge
    // repoints the pass (setMeshSource) whenever the edited mesh changes.
    lr::HeatmapPass heatmapPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
        .vertexBufferResourceName = sceneManager.selectedMeshHeatmapBufferName(),
        .indexBufferResourceName  = sceneManager.meshIndexBufferName(),
        .vertexBufferUploadResult = sceneManager.selectedMeshHeatmap(),
        .indexRange               = sceneManager.selectedMeshIndexRange(),
    });
    heatmapPass.build(viewer.frameGraph(), lr::SceneManager::selectedMeshHeatmapLayout());

    lr::AmbientOcclusionPass aoPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
        // World-space sampling radius. Nothing is loaded yet to scale it against, so this is a
        // starting value for the HBAO panel's slider rather than anything derived from the scene.
        .sphereRadius             = kDefaultAoSphereRadius,
    });
    aoPass.uploadResources(viewer.resources());
    aoPass.build(viewer.frameGraph());

    lr::PbrPass pbrPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
        .lightBufferResourceName  = sceneManager.lightBufferName(),
        .numLights                = sceneManager.numLights(),
        .pfMips                   = 8,
        .shadowImageResourceName = spotShadowPass.shadowImageName(),
        .shadowParamsBufferResourceName = spotShadowPass.paramsBufferName(),
        .cascadedShadowImageResourceName = cascadedShadowPass.shadowImageName(),
        .cascadedShadowParamsBufferResourceName = cascadedShadowPass.paramsBufferName(),
        .areaShadowImageResourceName = areaShadowPass.shadowImageName(),
        .areaShadowParamsBufferResourceName = areaShadowPass.paramsBufferName(),
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
        .spotShadowImageResourceName = spotShadowPass.shadowImageName(),
        .spotShadowParamsBufferResourceName = spotShadowPass.paramsBufferName(),
        .cascadedShadowImageResourceName = cascadedShadowPass.shadowImageName(),
        .cascadedShadowParamsBufferResourceName = cascadedShadowPass.paramsBufferName(),
        .areaShadowImageResourceName = areaShadowPass.shadowImageName(),
        .areaShadowParamsBufferResourceName = areaShadowPass.paramsBufferName(),
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
        transparentPass.setSceneGeometry(gpu.drawList(), gpu.geometryMeshes());
    };
    setTransparentGeometry(sceneManager.gpu());
    transparentPass.build(viewer.frameGraph(), lr::conventions::geometryMeshLayout());

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

    // As with the heatmap pass: one empty draw entry until EditorRenderBridge points it at a mesh.
    lr::OverlayPointsPass overlayPointsPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
        .pointsBufferResourceName = sceneManager.selectedMeshPointsBufferName(),
        .pointsBufferUploadResult = sceneManager.selectedMeshPoints(),
        .vertexCounts             = {0},
    });
    overlayPointsPass.build(viewer.frameGraph(), lr::SceneManager::selectedMeshPointsLayout());

    lr::FinalPass finalPass({
        .cameraBufferResourceName = sceneManager.cameraBufferName(),
        .swapchainFormat          = swapchainFormat,
        .showEnvironmentBackground = false,
    });
    finalPass.build(viewer.frameGraph());

    // Blender's selected-object outline, derived from the picking IDs. Blending into a LOAD
    // attachment counts as reading the swapchain, so the compiler orders this after every pass that
    // writes it — FinalPass included.
    lr::OutlinePass outlinePass({.outputFormat = swapchainFormat}, viewer.resources());
    outlinePass.build(viewer.frameGraph());

    // Editor interaction stays behind one facade while Engine owns it alongside the render pipeline.
    const lr::EditorRenderBridge::Passes editorPasses{
        .geometry       = &geometryPass,
        .transparent    = &transparentPass,
        .objectPicking  = &objectPickingPass,
        .spotShadow     = &spotShadowPass,
        .cascadedShadow = &cascadedShadowPass,
        .areaShadow     = &areaShadowPass,
        .heatmap        = &heatmapPass,
        .overlayPoints  = &overlayPointsPass,
        .overlayLines   = &overlayLinesPass,
        .outline        = &outlinePass,
    };
    lr::EditorRenderBridge editorRendering(sceneManager, editorPasses);
    lr::EditorSession editor(viewer, sceneManager, *camera, editorRendering);

    // -------------------------------------------------------------------------
    // Per-frame callbacks
    // -------------------------------------------------------------------------

    glm::vec3               environmentBackgroundColor(0.0f);
    bool                    showHdri = true;
    std::string             environmentLoadError;
    bool                    environmentDirty = false;
    std::string             sceneImportError;
    std::optional<fs::path> sceneDocumentPath;
    std::string             scenePersistenceError;
    lr::Settings            settings;

    auto showHdriSetting = Setting::bind("show_hdri", "showHDRI", SettingEditor::Checkbox, showHdri);
    auto background = Setting::bind("background_color", "Background color", SettingEditor::Color3,
                                    environmentBackgroundColor);
    background.enabled = [&] { return !showHdri || !scene.hdriPath().has_value(); };
    settings.add({
        .key         = "environment",
        .title       = "Environment",
        .defaultOpen = true,
        .settings    = {std::move(showHdriSetting), std::move(background)},
        .drawExtra   = [&] {
            ImGui::TextUnformatted("HDRI");
            ImGui::SameLine();
            if (scene.hdriPath()) ImGui::TextWrapped("%s", scene.hdriPath()->filename().string().c_str());
            else ImGui::TextDisabled("None (background color)");

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
            ImGui::TextDisabled("Supported format: Radiance HDR (.hdr)");
            if (!environmentLoadError.empty())
                ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Load failed: %s",
                                   environmentLoadError.c_str());
        },
        .onChanged = [&] {
            finalPass.setBackground(glm::vec4(environmentBackgroundColor, 1.0f),
                                    showHdri && scene.hdriPath().has_value());
        },
    });
    auto sphereRadius = Setting::bind("sphere_radius", "Sphere Radius", SettingEditor::Slider,
                                      aoPass.config().sphereRadius);
    sphereRadius.minimum = 0.0005f; sphereRadius.maximum = 0.2f; sphereRadius.format = "%.4f";
    sphereRadius.logarithmic = true;
    auto numSteps = Setting::bind("num_steps", "Num Steps", SettingEditor::Slider, aoPass.config().numSteps);
    numSteps.minimum = 1.0f; numSteps.maximum = 128.0f;
    auto numDirs = Setting::bind("num_directions", "Num Directions", SettingEditor::Slider, aoPass.config().numDirs);
    numDirs.minimum = 1.0f; numDirs.maximum = 128.0f;
    auto angleBias = Setting::bind("tan_angle_bias", "Tan Angle Bias", SettingEditor::Slider,
                                   aoPass.config().tanAngleBias);
    angleBias.minimum = 0.0f; angleBias.maximum = 1.0f;
    auto aoScalar = Setting::bind("ao_scalar", "AO Scalar", SettingEditor::Slider, aoPass.config().aoScalar);
    aoScalar.minimum = 0.0f; aoScalar.maximum = 5.0f;
    settings.add({.key = "hbao", .title = "HBAO",
                  .settings = {std::move(sphereRadius), std::move(numSteps), std::move(numDirs),
                               std::move(angleBias), std::move(aoScalar)},
                  .onChanged = [&] { aoPass.updateParams(viewer.resources()); }});

    auto shadowDistance = Setting::bind("shadow_distance", "Shadow Distance", SettingEditor::Drag,
                                        cascadedShadowPass.config().shadowDistance);
    shadowDistance.minimum = 1.0f; shadowDistance.maximum = 10000.0f; shadowDistance.speed = 1.0f;
    shadowDistance.format = "%.1f";
    auto splitLambda = Setting::bind("split_lambda", "Split Lambda", SettingEditor::Slider,
                                     cascadedShadowPass.config().splitLambda);
    splitLambda.minimum = 0.0f; splitLambda.maximum = 1.0f; splitLambda.format = "%.2f";
    auto depthPadding = Setting::bind("depth_padding", "Depth Padding", SettingEditor::Drag,
                                      cascadedShadowPass.config().depthPadding);
    depthPadding.minimum = 0.0f; depthPadding.maximum = 10000.0f; depthPadding.speed = 1.0f;
    depthPadding.format = "%.1f";
    settings.add({.key = "cascaded_shadows", .title = "Cascaded Shadows",
                  .settings = {std::move(shadowDistance), std::move(splitLambda), std::move(depthPadding)},
                  .drawExtra = [&] {
                      ImGui::TextDisabled("Cascades: %u", lr::CascadedShadowGpuData::cascadeCount);
                      ImGui::TextDisabled("Resolution: %u x %u per cascade", cascadedShadowPass.config().resolution,
                                          cascadedShadowPass.config().resolution);
                      ImGui::TextDisabled("Cascade count and resolution require a framegraph rebuild.");
                  }});
    // Constructed after everything its callbacks capture, so all registrations disconnect first.
    std::vector<lr::CallbackConnection> appConnections;

    // Whenever SceneGpu re-packs geometry (an import, lights added or removed) or re-uploads the lights,
    // keep the passes' draw lists and light count in step.
    appConnections.push_back(sceneManager.gpu().onGeometryRebuilt([&](const lr::SceneGpu &gpu) {
        const lr::SceneDrawList draws = gpu.drawList();
        geometryPass.setSceneGeometry(draws);
        objectPickingPass.setSceneGeometry(draws);
        spotShadowPass.setSceneGeometry(draws);
        spotShadowPass.setLightObjects(gpu.lightObjects());
        cascadedShadowPass.setSceneGeometry(draws);
        cascadedShadowPass.setLightObjects(gpu.lightObjects());
        areaShadowPass.setSceneGeometry(draws);
        areaShadowPass.setLightObjects(gpu.lightObjects());
        setTransparentGeometry(gpu);
    }));
    appConnections.push_back(sceneManager.gpu().onLightsUploaded([&](uint32_t numLights) {
        pbrPass.setNumLights(numLights);
        transparentPass.setNumLights(numLights);
    }));
    appConnections.push_back(scene.registerObjectsDestroyedCallback([&](std::span<const lr::SceneObjectId>) {
        physicsWorld.onSceneChanged();
    }));

    appConnections.push_back(viewer.onGui([&]() {
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

        drawSettings(settings);

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
        // Added after the tooltip above, which reads the last item and would otherwise attach to
        // this button instead of the disabled one it explains.
        ImGui::SameLine();
        if (ImGui::Button("Add Scene Object"))
        {
            // Empty but for a transform, which every other component reads as its object's place in
            // the world — so the object is ready to receive one straight away. What it becomes is
            // then decided in the Inspector.
            lr::SceneObject &emptyObject = scene.createSceneObject();
            emptyObject.name             = "Scene Object";
            emptyObject.addComponent<lr::TransformComponent>();
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
        editor.drawAnimationWindow();
    }));

    // Application-level rendering policy: the C++ demo owns the IBL shaders, resource names,
    // and rebuild timing. Scene only retains the authored HDRI path for persistence.
    appConnections.push_back(viewer.onLateUpdate([&](float, VkExtent2D) {
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
                                    showHdri && scene.hdriPath().has_value());

            environmentLoadError.clear();
        } catch (const std::exception &e)
        {
            environmentLoadError = e.what();
            spdlog::error("Failed to update environment: {}", e.what());
        }

        environmentDirty = false;
    }));

    appConnections.push_back(viewer.onUpdate([&](float dt, VkExtent2D) {
        sceneManager.animationSystem().update(dt);
        physicsWorld.update(dt);
    }));

    appConnections.push_back(viewer.onUpdate([&](float dt, VkExtent2D extent) {
        cameraController.update(viewer.input(), dt, editor.allowsViewportNavigation());
        cascadedShadowPass.setViewportExtent(extent);
    }));

    // Registers SceneManager's own onUpdate (aspect tracking) and onLateUpdate (flushDirty —
    // runs after every onUpdate above, so it sees the results of this frame's camera
    // controller / gizmo / GUI edits and does at most one GPU re-upload per dirtied resource
    // rather than one per individual mutation) callbacks — see SceneManager::registerCallbacks.
    sceneManager.registerCallbacks(viewer);

    viewer.addImguiPass();
    viewer.run();
}

} // namespace lr
