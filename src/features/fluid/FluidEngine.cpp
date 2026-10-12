#include "features/fluid/FluidEngine.hpp"

#include "core/app/Viewer.hpp"
#include "core/editor/camera/SphericalCameraController.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"
#include "core/upload/CameraUploader.hpp"
#include "features/fluid/FluidRenderer.hpp"
#include "features/fluid/FluidSimulation.hpp"
#include "features/fluid/ParticleBillboardPass.hpp"

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
    // GPU simulation with an exact 50 x 56 x 50 lattice: 140,000 particles, matching the
    // large real-time examples used to evaluate PBF-scale behaviour.
    // -------------------------------------------------------------------------
    FluidSimulation simulation({.sourceDimensions = {50, 56, 50},
                                .maxParticles     = 140000,
                                .startPaused      = false},
                               viewer.resources());
    spdlog::info("fluid: {} GPU particles", simulation.particleCount());

    FluidRenderer renderer({.cameraBufferResourceName   = cameraUploader.bufferName(),
                            .swapchainFormat            = viewer.swapchainFormat(),
                            .maxParticleCount           = simulation.capacity(),
                            .instanceBufferResourceName = simulation.particleBufferName(),
                            .instanceStride             = sizeof(ParticleGpuData)},
                           viewer.resources());

    simulation.build(viewer.frameGraph());
    renderer.build(viewer.frameGraph());

    // Frame the simulation domain so the fluid is on screen without the user having to orbit to
    // find it.
    const glm::vec3 center = (simulation.boundsMin() + simulation.boundsMax()) * 0.5f;
    cameraController.setOrbitState({
        .target    = center,
        .radius    = glm::length(simulation.boundsMax() - simulation.boundsMin()) * 1.8f,
        .azimuth   = 0.6f,
        .elevation = 0.35f,
    });

    // The box is static, so it is uploaded once rather than every frame.
    std::vector<FluidRenderer::Line> lines;
    FluidRenderer::appendBoxOutline(lines, simulation.boundsMin(), simulation.boundsMax(),
                                    glm::vec3(0.45f, 0.45f, 0.5f));
    renderer.setLines(lines);

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

        simulation.update(dt);
        // The instance records themselves are written by the GPU solver's publish pass.
        renderer.setParticleCount(simulation.particleCount(), simulation.particleColor());
    }));

    connections.push_back(viewer.onGui([&]() {
        // Placed explicitly so the panel starts on screen rather than wherever a stale imgui.ini
        // left it.
        ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340.0f, 300.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Fluid");
        simulation.onGUI();
        ImGui::End();
    }));

    viewer.addImguiPass();
    viewer.run();
}

} // namespace lr
