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
#include "features/fluid/FluidSolverGpu.hpp"
#include "features/fluid/ParticleBillboardPass.hpp"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <vector>

namespace lr
{
namespace
{

// Simulation setup shared by both solvers, so switching between them changes where the solve runs
// and nothing else. Spacing is the source of truth: the smoothing radius, rest density and drawn
// particle size all derive from it, which is what keeps the sim from being set up inconsistently.
struct SimSetup
{
    float     spacing = 0.05f;
    float     smoothingRadius() const { return spacing * 2.0f; }
    float     particleRadius() const { return spacing * 0.9f * 0.5f; }
    glm::vec3 boundsMin{-0.5f, 0.0f, -0.5f};
    glm::vec3 boundsMax{0.5f, 1.0f, 0.5f};
    glm::vec3 sourceFraction{0.4f, 0.6f, 0.4f};
    glm::vec3 color{0.25f, 0.5f, 0.95f};
    int       solverIterations = 8;
    float     fixedDeltaTime   = 1.0f / 60.0f;
    int       maxStepsPerFrame = 4;
};

std::vector<glm::vec3> makeSourceBlock(const SimSetup &setup)
{
    // Inset by one spacing so no particle starts exactly on a wall, where the boundary clamp would
    // fight the density solve on the very first step.
    const glm::vec3 extent = setup.boundsMax - setup.boundsMin;
    const glm::vec3 min    = setup.boundsMin + glm::vec3(setup.spacing);
    const glm::vec3 max    = min + extent * setup.sourceFraction;
    return FluidSolverCpu::makeParticleBlock(min, max, setup.spacing);
}

FluidSolverGpu::Config gpuConfig(const SimSetup &setup, uint32_t capacity)
{
    const float radius = setup.smoothingRadius();

    FluidSolverGpu::Config config;
    config.smoothingRadius = radius;
    config.particleMass    = 1.0f;
    // Derived from the lattice the particles actually start on rather than chosen independently,
    // which is what keeps the block from inflating or collapsing on the first step.
    config.restDensity      = FluidSolverCpu::restDensityForSpacing(setup.spacing, radius, 1.0f);
    config.relaxation       = config.restDensity * 0.01f;
    config.solverIterations = setup.solverIterations;
    // Headroom above the configured count so iterations can be raised from the GUI without
    // rebuilding the frame graph.
    config.maxSolverIterations = std::max(setup.solverIterations, 16);
    config.boundsMin           = setup.boundsMin;
    config.boundsMax           = setup.boundsMax;
    config.maxParticles        = capacity;
    config.fixedDeltaTime      = setup.fixedDeltaTime;
    config.maxStepsPerFrame    = setup.maxStepsPerFrame;
    config.startPaused         = true;
    config.particleRadius      = setup.particleRadius();
    config.particleColor       = setup.color;
    return config;
}

} // namespace

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
    // Simulation. Both solvers are built: the CPU one is the reference the GPU solve was written
    // against, and keeping them side by side here is what makes it possible to switch at runtime
    // and see that they behave the same. The GPU solver is the default -- it is the one that scales.
    // -------------------------------------------------------------------------
    const SimSetup               setup;
    const std::vector<glm::vec3> source = makeSourceBlock(setup);

    FluidDemo fluidDemo({
        .particleSpacing      = setup.spacing,
        .smoothingRadiusScale = 2.0f,
        .solverIterations     = setup.solverIterations,
        .boundsMin            = setup.boundsMin,
        .boundsMax            = setup.boundsMax,
        .sourceFraction       = setup.sourceFraction,
        .fixedDeltaTime       = setup.fixedDeltaTime,
        .maxStepsPerFrame     = setup.maxStepsPerFrame,
        .startPaused          = true,
        .particleColor        = setup.color,
    });

    FluidSolverGpu gpuSolver(gpuConfig(setup, static_cast<uint32_t>(source.size())), viewer.resources());
    gpuSolver.setParticles(source);

    bool useGpu = true;

    // -------------------------------------------------------------------------
    // Renderer. Its instance data is the GPU solver's published particle buffer, so on the GPU path
    // the solved positions go from the compute passes to the rasteriser without touching the host.
    // The CPU path writes the same records through setParticles() on its own buffer, so both are
    // drawn by the same pass -- which is also why the two can be compared by eye at all.
    // -------------------------------------------------------------------------
    FluidRenderer gpuRenderer({.cameraBufferResourceName   = cameraUploader.bufferName(),
                               .swapchainFormat            = viewer.swapchainFormat(),
                               .maxParticleCount           = static_cast<uint32_t>(source.size()),
                               .instanceBufferResourceName = gpuSolver.config().particleBufferResourceName,
                               .instanceStride             = sizeof(ParticleGpuData)},
                              viewer.resources());
    FluidRenderer cpuRenderer({.cameraBufferResourceName = cameraUploader.bufferName(),
                               .swapchainFormat          = viewer.swapchainFormat(),
                               .resourceNamePrefix       = "cpu_"},
                              viewer.resources());

    gpuSolver.build(viewer.frameGraph());
    gpuRenderer.build(viewer.frameGraph());
    cpuRenderer.build(viewer.frameGraph());

    // Frame the simulation domain so the fluid is on screen without the user having to orbit to
    // find it.
    const glm::vec3 center = (setup.boundsMin + setup.boundsMax) * 0.5f;
    cameraController.setOrbitState({
        .target    = center,
        .radius    = glm::length(setup.boundsMax - setup.boundsMin) * 1.8f,
        .azimuth   = 0.6f,
        .elevation = 0.35f,
    });

    // The box is static, so it is uploaded once rather than every frame. Both renderers draw it so
    // the domain stays visible whichever solver is running.
    std::vector<FluidRenderer::Line> lines;
    FluidRenderer::appendBoxOutline(lines, setup.boundsMin, setup.boundsMax, glm::vec3(0.45f, 0.45f, 0.5f));
    gpuRenderer.setLines(lines);
    cpuRenderer.setLines({});

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

        if (useGpu)
        {
            gpuSolver.update(dt);
            // Only a count: the records themselves are written by the solver's publish pass.
            gpuRenderer.setParticleCount(gpuSolver.particleCount(), gpuSolver.config().particleColor);
            cpuRenderer.setParticleCount(0, setup.color);
        } else
        {
            fluidDemo.update(dt);
            const auto &config = fluidDemo.config();
            const float radius = config.particleSpacing * config.particleRenderScale * 0.5f;
            cpuRenderer.setParticles(fluidDemo.solver().positions(), radius, config.particleColor);
            gpuRenderer.setParticleCount(0, setup.color);
        }
    }));

    connections.push_back(viewer.onGui([&]() {
        // Placed explicitly so the panel starts on screen rather than wherever a stale imgui.ini
        // left it.
        ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340.0f, 320.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Fluid");

        if (ImGui::RadioButton("GPU (compute)", useGpu))
        {
            useGpu = true;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("CPU (reference)", !useGpu))
        {
            useGpu = false;
        }
        ImGui::Separator();

        if (useGpu)
        {
            bool paused = gpuSolver.paused();
            if (ImGui::Button(paused ? "Play" : "Pause"))
            {
                gpuSolver.setPaused(!paused);
            }
            ImGui::SameLine();
            if (ImGui::Button("Step"))
            {
                gpuSolver.stepOnce();
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset"))
            {
                gpuSolver.setParticles(source);
            }

            ImGui::Text("Particles: %u", gpuSolver.particleCount());
            ImGui::Text("Steps this frame: %u", gpuSolver.pendingSteps());

            // Capped at what build() declared passes for, so raising it past that does nothing.
            int iterations = gpuSolver.config().solverIterations;
            if (ImGui::DragInt("Iterations", &iterations, 1.0f, 0, gpuSolver.config().maxSolverIterations))
            {
                gpuSolver.config().solverIterations = std::clamp(iterations, 0, gpuSolver.config().maxSolverIterations);
                gpuSolver.updateParams();
            }
            float viscosity = gpuSolver.config().viscosity;
            if (ImGui::DragFloat("Viscosity", &viscosity, 0.005f, 0.0f, 0.5f, "%.3f"))
            {
                gpuSolver.config().viscosity = viscosity;
                gpuSolver.updateParams();
            }
            float wallDensity = gpuSolver.config().boundsDensity;
            if (ImGui::DragFloat("Wall Density", &wallDensity, 0.02f, 0.0f, 2.0f, "%.2f"))
            {
                gpuSolver.config().boundsDensity = wallDensity;
                gpuSolver.updateParams();
            }
            float friction = gpuSolver.config().boundsFriction;
            if (ImGui::DragFloat("Wall Friction", &friction, 0.005f, 0.0f, 1.0f, "%.3f"))
            {
                gpuSolver.config().boundsFriction = friction;
                gpuSolver.updateParams();
            }
            ImGui::TextWrapped("Densities read back through the GPU buffer would stall the "
                               "pipeline, so the convergence readout lives in the CPU path and in "
                               "the fluid_solver_gpu_smoke test.");
        } else
        {
            fluidDemo.onGUI();
        }

        ImGui::End();
    }));

    viewer.addImguiPass();
    viewer.run();
}

} // namespace lr
