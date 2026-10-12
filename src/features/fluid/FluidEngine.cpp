#include "features/fluid/FluidEngine.hpp"

#include "core/app/Viewer.hpp"
#include "core/editor/camera/SphericalCameraController.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"
#include "core/upload/CameraUploader.hpp"
#include "features/fluid/FluidDemo.hpp"
#include "features/fluid/FluidRenderer.hpp"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <vector>

namespace lr
{

void FluidEngine::run()
{
    spdlog::set_level(spdlog::level::debug);

    Viewer viewer({.title = "lr — fluid"});

    // -------------------------------------------------------------------------
    // Camera. A Scene with a single camera object is the smallest thing that satisfies
    // CameraUploader, which reads the camera's transform through its owning object. No
    // SceneManager, materials or lights are involved.
    // -------------------------------------------------------------------------
    Scene        scene;
    SceneObject &cameraObject = scene.createSceneObject();
    cameraObject.name         = "Camera";
    cameraObject.addComponent<Camera>();
    cameraObject.addComponent<TransformComponent>();
    auto &cameraController = cameraObject.addComponent<SphericalCameraController>();

    CameraUploader cameraUploader(viewer.resources());

    // -------------------------------------------------------------------------
    // Simulation and renderer
    // -------------------------------------------------------------------------
    FluidDemo fluidDemo({});

    FluidRenderer fluidRenderer({.cameraBufferResourceName = cameraUploader.bufferName(),
                                 .swapchainFormat          = viewer.swapchainFormat()},
                                viewer.resources());
    fluidRenderer.build(viewer.frameGraph());

    // Frame the simulation domain so the fluid is on screen without the user having to orbit to
    // find it.
    const glm::vec3 boundsMin = fluidDemo.config().boundsMin;
    const glm::vec3 boundsMax = fluidDemo.config().boundsMax;
    const glm::vec3 center    = (boundsMin + boundsMax) * 0.5f;
    cameraController.setOrbitState({
        .target    = center,
        .radius    = glm::length(boundsMax - boundsMin) * 1.8f,
        .azimuth   = 0.6f,
        .elevation = 0.35f,
    });

    // The box is static, so it is uploaded once rather than every frame.
    std::vector<FluidRenderer::Line> lines;
    FluidRenderer::appendBoxOutline(lines, boundsMin, boundsMax, glm::vec3(0.45f, 0.45f, 0.5f));
    fluidRenderer.setLines(lines);

    // -------------------------------------------------------------------------
    // Per-frame
    // -------------------------------------------------------------------------
    std::vector<CallbackConnection> connections;

    connections.push_back(viewer.onUpdate([&](float dt, VkExtent2D extent) {
        // The GUI owns the pointer only while ImGui wants it; otherwise orbiting is allowed.
        const bool navigationAllowed = !ImGui::GetIO().WantCaptureMouse;
        cameraController.update(viewer.input(), dt, navigationAllowed);

        const float aspect =
            (extent.height == 0) ? 1.0f : static_cast<float>(extent.width) / static_cast<float>(extent.height);
        cameraUploader.upload(cameraObject, aspect);

        fluidDemo.update(dt);

        const auto &config = fluidDemo.config();
        // Matches FluidDemo's own sizing: the mesh is a unit-radius sphere, so this is a radius.
        const float radius = config.particleSpacing * config.particleRenderScale * 0.5f;
        fluidRenderer.setParticles(fluidDemo.solver().positions(), radius, config.particleColor);
    }));

    connections.push_back(viewer.onGui([&]() {
        // Placed explicitly so the panel starts on screen rather than wherever a stale imgui.ini
        // left it.
        ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(320.0f, 260.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Fluid");
        fluidDemo.onGUI();
        ImGui::End();
    }));

    viewer.addImguiPass();
    viewer.run();
}

} // namespace lr
