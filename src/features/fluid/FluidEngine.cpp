#include "features/fluid/FluidEngine.hpp"

#include "core/app/Viewer.hpp"
#include "core/editor/camera/SphericalCameraController.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"
#include "core/upload/CameraUploader.hpp"
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

// Simulation setup. Spacing is the source of truth: the smoothing radius, rest density and drawn
// particle size all derive from it, so the sim cannot be set up inconsistently.
struct SimSetup
{
    // Fixed to roughly 140k particles for the standalone GPU demo. The source volume and lattice
    // discretisation produce 137,700 particles at this spacing.
    float spacing = 0.0089f;
    // Multiple of spacing used as the kernel support radius. Around 2 gives each particle the
    // ~30-40 neighbours SPH needs to estimate density well.
    float smoothingRadius() const { return spacing * 2.0f; }
    // Drawn sphere diameter as a multiple of spacing. Below 1 leaves visible gaps; above 1 overlaps
    // into a solid mass.
    float     particleRenderScale = 0.9f;
    float     particleRadius() const { return spacing * particleRenderScale * 0.5f; }
    glm::vec3 boundsMin{-0.5f, 0.0f, -0.5f};
    glm::vec3 boundsMax{0.5f, 1.0f, 0.5f};
    // Fraction of the domain the initial block of fluid occupies, per axis.
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
    // which is what keeps the block from inflating or collapsing on the first step. This is why
    // changing spacing has to go through here rather than being set on the solver directly: the
    // smoothing radius, rest density and relaxation all follow from it.
    config.restDensity      = FluidSolverCpu::restDensityForSpacing(setup.spacing, radius, 1.0f);
    config.relaxation       = config.restDensity * 0.01f;
    config.solverIterations = setup.solverIterations;
    // Headroom above the configured count so iterations can be raised from the GUI without
    // rebuilding the frame graph, which fixes how many passes are declared.
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
    // Simulation and renderer.
    //
    // The solve runs entirely in compute passes, and the renderer's per-instance data is the
    // solver's published particle buffer, so solved positions reach the rasteriser without touching
    // the host. Those records start with a vec4 of centre and radius, which is exactly what the
    // instance attribute reads; the stride steps over the colour that follows.
    //
    // FluidSolverCpu remains the reference the GPU solve is diffed against, but that comparison
    // lives in fluid_solver_gpu_smoke, which can read buffers back without stalling a frame loop.
    // Drawing it here would take a second FluidRenderer, and two of them clear the same swapchain
    // image -- whichever ran second blanked the other's particles, which is exactly what left this
    // viewer showing nothing when it was tried.
    // -------------------------------------------------------------------------
    SimSetup setup;

    const std::vector<glm::vec3> source = makeSourceBlock(setup);
    const uint32_t capacity = static_cast<uint32_t>(source.size());
    spdlog::info("fluid: {} particles at fixed spacing {:.4f}", source.size(), setup.spacing);

    FluidSolverGpu solver(gpuConfig(setup, capacity), viewer.resources());
    solver.setParticles(source);

    FluidRenderer renderer({.cameraBufferResourceName   = cameraUploader.bufferName(),
                            .swapchainFormat            = viewer.swapchainFormat(),
                            .maxParticleCount           = capacity,
                            .instanceBufferResourceName = solver.config().particleBufferResourceName,
                            .instanceStride             = sizeof(ParticleGpuData)},
                           viewer.resources());

    solver.build(viewer.frameGraph());
    renderer.build(viewer.frameGraph());

    // Frame the simulation domain so the fluid is on screen without the user having to orbit to
    // find it.
    const glm::vec3 center = (setup.boundsMin + setup.boundsMax) * 0.5f;
    cameraController.setOrbitState({
        .target    = center,
        .radius    = glm::length(setup.boundsMax - setup.boundsMin) * 1.8f,
        .azimuth   = 0.6f,
        .elevation = 0.35f,
    });

    // The box is static, so it is uploaded once rather than every frame.
    std::vector<FluidRenderer::Line> lines;
    FluidRenderer::appendBoxOutline(lines, setup.boundsMin, setup.boundsMax, glm::vec3(0.45f, 0.45f, 0.5f));
    renderer.setLines(lines);

    // Restore the fixed source lattice while preserving the current live solver controls.
    const auto resetSimulation = [&] {
        const bool wasPaused = solver.paused();
        solver.setParticles(source);
        solver.setPaused(wasPaused);
    };

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

        solver.update(dt);
        // Only a count: the instance records themselves are written by the solver's publish pass.
        renderer.setParticleCount(solver.particleCount(), solver.config().particleColor);
    }));

    connections.push_back(viewer.onGui([&]() {
        // Placed explicitly so the panel starts on screen rather than wherever a stale imgui.ini
        // left it.
        ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340.0f, 300.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Fluid");

        const bool paused = solver.paused();
        if (ImGui::Button(paused ? "Play" : "Pause"))
        {
            solver.setPaused(!paused);
        }
        ImGui::SameLine();
        if (ImGui::Button("Step"))
        {
            solver.stepOnce();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
        {
            resetSimulation();
        }

        ImGui::Text("Particles: %u / %u", solver.particleCount(), capacity);
        ImGui::Text("Steps this frame: %u", solver.pendingSteps());

        auto &config = solver.config();

        // Capped at what build() declared passes for, so raising it past that would do nothing.
        int iterations = config.solverIterations;
        if (ImGui::DragInt("Iterations", &iterations, 1.0f, 0, config.maxSolverIterations))
        {
            config.solverIterations = std::clamp(iterations, 0, config.maxSolverIterations);
            solver.updateParams();
        }
        // Energy has to leave the system somewhere: with viscosity at zero the positional solve
        // feeds its own corrections back into velocity and a dropped block never settles.
        if (ImGui::DragFloat("Viscosity", &config.viscosity, 0.005f, 0.0f, 0.5f, "%.3f"))
        {
            solver.updateParams();
        }
        // At zero the walls contribute no density, so particles beside one read as free surface and
        // get extruded up the corners by the denser fluid behind them.
        if (ImGui::DragFloat("Wall Density", &config.boundsDensity, 0.02f, 0.0f, 2.0f, "%.2f"))
        {
            solver.updateParams();
        }
        if (ImGui::DragFloat("Wall Friction", &config.boundsFriction, 0.005f, 0.0f, 1.0f, "%.3f"))
        {
            solver.updateParams();
        }
        ImGui::Separator();

        // Purely visual: the drawn sphere size, which the publish pass writes into each record.
        if (ImGui::DragFloat("Particle Size", &setup.particleRenderScale, 0.01f, 0.1f, 2.0f, "%.2f"))
        {
            config.particleRadius = setup.particleRadius();
        }

        // Read by the publish pass as a push constant, so there is no uniform to refresh.
        ImGui::ColorEdit3("Color", &config.particleColor.x);

        ImGui::TextWrapped("Reading densities back would stall the pipeline, so the convergence "
                           "readout lives in the fluid_solver_gpu_smoke test.");

        ImGui::End();
    }));

    viewer.addImguiPass();
    viewer.run();
}

} // namespace lr
