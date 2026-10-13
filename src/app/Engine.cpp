#include "Engine.hpp"
#include "app/EditorSession.hpp"
#include "app/SceneDocument.hpp"
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
#include "core/passes/shadow/PointShadowPass.hpp"
#include "core/passes/transparent/TransparentPass.hpp"
#include "core/passes/ambientocclusion/AmbientOcclusionPass.hpp"
#include "core/passes/overlaygeometry/OverlayGeometryPass.hpp"
#include "core/passes/objectpicking/ObjectPickingPass.hpp"
#include "core/passes/outline/OutlinePass.hpp"
#include "core/passes/overlaylines/OverlayLinesPass.hpp"
#include "core/passes/overlaypoints/OverlayPointsPass.hpp"

#include "core/scene/AreaLightVisual.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/editor/camera/SphericalCameraController.hpp"
#include "core/editor/ErrorReporter.hpp"
#include "core/scene/EngineConventions.hpp"
#include "core/scene/SceneManager.hpp"
#include "core/scene/Scene.hpp"

#include "features/rigid_body/PhysicsWorld.hpp"
#if defined(LR_ENABLE_CUDA)
#include "features/eulerian_particle/SmokeSimulation.hpp"
#endif

#include <ImGuiFileDialog.h>
#include <imgui.h>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <spdlog/spdlog.h>

#include <cmath>
#include <exception>
#include <filesystem>
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

    lr::Viewer::Config viewerConfig{.title = "lr"};
#if defined(LR_ENABLE_CUDA)
    viewerConfig.enableExternalSemaphoreInterop = true;
#endif
    lr::Viewer viewer(viewerConfig);

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
            {"ibl_brdf_lut", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
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

#if defined(LR_ENABLE_CUDA)
    // Grid-based smoke on CUDA. The simulation volumes are persistent registry images that CUDA
    // writes through exported memory; the simulation's own ray-march pass then samples the
    // published density image in the normal frame graph. It owns all of its own settings and
    // draws them in the "Smoke" window, so nothing here configures it per-knob.
    // The volumes are allocated at maxCellsPerAxis on every axis, so the grid below can be resized
    // in the editor without any reallocation. The box is authored in cells and scaled into world
    // space by simulationScale, so scale and position are free to change mid-simulation.
    lr::eulerian::SmokeSimulation smokeSimulation(
        {
            .solver = {.jacobiIterations = 40,
                       .step             = {.buoyancy    = 8.0f,
                                            .densityDamp = 0.15f,
                                            .vorticity   = 2.0f},
                       .source           = {.density = 18.0f, .velocity = {0.0f, 5.0f, 0.0f}}},
            .simulationScale          = 2.0f / 64.0f,
            .gridSize                 = {64, 96, 64},
            // The box is centred here, so at 64 x 96 x 64 cells of 2/64 units it spans
            // 2 x 3 x 2 world units and sits on the ground plane.
            .boundsCentre             = {0.0f, 1.5f, 0.0f},
            .maxCellsPerAxis          = 100,
            .cameraBufferResourceName = sceneManager.cameraBufferName(),
            .outputFormat             = swapchainFormat,
            .sourcePosition           = {0.0f, 0.45f, 0.0f},
            .sourceRadius             = 0.25f,
        },
        viewer.resources(), viewer.context());
#endif

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

    lr::PointShadowPass pointShadowPass(viewer.resources(), {
        .geometry = sceneManager.gpu().geometryPassConfig(),
        .lightObjects = sceneManager.gpu().lightObjects(),
    });
    pointShadowPass.build(viewer.frameGraph(), lr::conventions::geometryMeshLayout());

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
        .pointShadowImageResourceName = pointShadowPass.shadowImageName(),
        .pointShadowParamsBufferResourceName = pointShadowPass.paramsBufferName(),
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
        .pointShadowImageResourceName = pointShadowPass.shadowImageName(),
        .pointShadowParamsBufferResourceName = pointShadowPass.paramsBufferName(),
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

#if defined(LR_ENABLE_CUDA)
    // Blends the ray-marched volume over the completed scene but before selection outlines and
    // ImGui. The pass also samples the opaque depth buffer to stop rays at scene geometry.
    smokeSimulation.build(viewer.frameGraph());
#endif

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
        .pointShadow    = &pointShadowPass,
        .cascadedShadow = &cascadedShadowPass,
        .areaShadow     = &areaShadowPass,
        .heatmap        = &heatmapPass,
        .overlayGeometry = &overlayGeometryPass,
        .overlayPoints  = &overlayPointsPass,
        .overlayLines   = &overlayLinesPass,
        .outline        = &outlinePass,
    };
    lr::EditorRenderBridge editorRendering(sceneManager, editorPasses);
#if defined(LR_ENABLE_CUDA)
    // The simulation box is not a scene object, so it contributes its outline through the bridge's
    // producer hook rather than through the scene's collider and gizmo overlays.
    editorRendering.addOverlayLineProducer(
        [&smokeSimulation](lr::OverlayLineBuilder &builder) { smokeSimulation.appendOverlayLines(builder); });
#endif
    lr::ErrorReporter      errorReporter;
    lr::EditorSession editor(viewer, sceneManager, *camera, editorRendering, errorReporter);

    // -------------------------------------------------------------------------
    // Per-frame callbacks
    // -------------------------------------------------------------------------

    glm::vec3               environmentBackgroundColor(0.0f);
    bool                    showHdri = true;
    std::string             environmentLoadError;
    bool                    environmentDirty = false;
    lr::Settings            settings;

    // Owns the scene document's path, its load/save/import sequencing, and the errors from it.
    lr::SceneDocument sceneDocument(viewer, sceneManager, editor, physicsWorld, sceneLoadConfig, errorReporter);
    // A loaded scene brings its own authored HDRI path, so the environment has to be rebuilt from
    // it. Which passes that touches stays here — see the onLateUpdate below.
    sceneDocument.setSceneLoadedCallback([&environmentDirty] { environmentDirty = true; });

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
        pointShadowPass.setSceneGeometry(draws);
        pointShadowPass.setLightObjects(gpu.lightObjects());
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
                    sceneDocument.openLoadDialog();
                }
                if (ImGui::MenuItem("Save", nullptr, false, sceneDocument.path().has_value()))
                {
                    sceneDocument.save();
                }
                if (ImGui::MenuItem("Save As..."))
                {
                    sceneDocument.openSaveAsDialog();
                }
                ImGui::Separator();
                // Geometry only, hence the wording: the .obj carries every rendered object's
                // positions and faces, not their normals, uvs or materials.
                if (ImGui::MenuItem("Export Scene Geometry (.obj)...", nullptr, false,
                                    sceneDocument.canExportGeometry()))
                {
                    sceneDocument.openExportGeometryDialog();
                }
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                {
                    ImGui::SetTooltip("Writes every rendered object's vertices and faces, placed in world "
                                      "space and in their current pose - no normals, UVs or materials.");
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

#if defined(LR_ENABLE_CUDA)
        // Its own window rather than a Settings category: the solver has enough knobs, plus
        // play/pause/step controls, that it reads better with room of its own.
        ImGui::SetNextWindowPos(ImVec2(topLeft.x + panelSize.x + 16.0f, topLeft.y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(panelSize.x, viewport->WorkSize.y * 0.6f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Smoke");
        smokeSimulation.onGUI();
        ImGui::End();
#endif

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
        editor.drawScenePanel(sceneDocument);
        ImGui::End();

        // Outside every window above: a modal cannot be opened from inside one.
        sceneDocument.drawDialogs();

        ImGui::SetNextWindowPos(bottomRight, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(panelSize, ImGuiCond_FirstUseEver);
        ImGui::Begin("Features");
        editor.drawFeaturePanel();
        ImGui::End();
        editor.drawTransformWindow();
        editor.drawAnimationWindow();
        editor.drawSkinningWindow();
        errorReporter.drawPopup();
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
                {"ibl_brdf_lut", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                {"ibl_irradiance", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                {"ibl_prefiltered", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            });

            finalPass.setBackground(glm::vec4(environmentBackgroundColor, 1.0f),
                                    showHdri && scene.hdriPath().has_value());

            environmentLoadError.clear();
        } catch (const std::exception &e)
        {
            environmentLoadError = e.what();
            errorReporter.report("Could not update environment", e.what());
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

#if defined(LR_ENABLE_CUDA)
    // Queues the external-memory handshake before this frame's frame-graph submission. The calls do
    // not wait on the host: queue order is Vulkan release -> CUDA step -> Vulkan acquire -> render.
    appConnections.push_back(viewer.onLateUpdate([&](float dt, VkExtent2D) {
        smokeSimulation.update(dt);
    }));
#endif

    viewer.addImguiPass();
    viewer.run();
}

} // namespace lr
