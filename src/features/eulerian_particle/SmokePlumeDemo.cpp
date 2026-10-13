#include "features/eulerian_particle/SmokeSolver.hpp"

#include "core/Paths.hpp"
#include "core/app/Viewer.hpp"
#include "core/upload/CameraUploader.hpp"
#include "features/eulerian_particle/SmokeRenderPass.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace
{

constexpr const char *kCameraBuffer = "smoke_demo_camera";

// The simulation grid. It is deliberately taller than it is wide, which is what makes this demo
// useful: the render bounds below have to carry the same 1.5:1 aspect or the plume renders
// squashed, since the shader maps the box to the volume with a plain normalized divide.
constexpr VkExtent3D kGridExtent{48, 72, 48};

lr::CameraGpuData makeCamera(float aspect)
{
    const glm::vec3 eye{3.4f, 2.25f, 4.4f};
    const glm::vec3 target{0.0f, 1.25f, 0.0f};

    lr::CameraGpuData camera{};
    camera.view = glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
    camera.proj = glm::perspective(glm::radians(50.0f), std::max(aspect, 0.001f), 0.1f, 100.0f);
    camera.proj[1][1] *= -1.0f;

    // GLM's perspective matrix uses OpenGL's [-1, 1] depth. Match Camera::projectionMatrix and
    // remap it to Vulkan's [0, 1] range.
    glm::mat4 clipFix(1.0f);
    clipFix[2][2] = 0.5f;
    clipFix[3][2] = 0.5f;
    camera.proj   = clipFix * camera.proj;

    camera.viewProj = camera.proj * camera.view;
    camera.invView  = glm::inverse(camera.view);
    camera.invProj  = glm::inverse(camera.proj);
    camera.position = eye;
    return camera;
}

int requestedFrameCount(int argc, char **argv)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string_view(argv[i]) == "--frames") return std::max(std::atoi(argv[i + 1]), 0);
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        lr::Viewer viewer({
            .title                           = "CUDA smoke plume",
            .width                           = 1280,
            .height                          = 800,
            .enableValidation                = true,
            .enableGui                       = false,
            .enableExternalSemaphoreInterop  = true,
        });

        viewer.resources().registerDynamicBuffer(kCameraBuffer, sizeof(lr::CameraGpuData),
                                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        lr::eulerian::SmokeSolver::Config simulation{};
        simulation.extent             = kGridExtent;
        simulation.jacobiIterations   = 140;
        simulation.step.buoyancy      = 0.1f;
        simulation.step.densityDamp   = 0.16f;
        simulation.step.velocityDamp  = 0.04f;
        simulation.step.vorticity     = 1.5f;
        simulation.source.radius      = 0.09f;
        simulation.source.density     = 16.0f;
        simulation.source.velocity[1] = 3.5f;

        lr::eulerian::SmokeSolver solver(viewer.resources(), viewer.context(), simulation);

        // A procedural opaque scene, so the smoke/depth interaction is visible without any
        // external assets. The smoke pass samples this depth buffer to stop rays at geometry.
        lr::FrameGraph &graph = viewer.frameGraph();
        graph.addPass("smoke_demo_scene")
            .type(lr::PassType::Fullscreen)
            .vertShader((lr::paths::shaderDirectory() / "fullscreen.vert.spv").string())
            .fragShader((lr::paths::shaderDirectory() / "smoke_demo_scene.frag.spv").string())
            .uniformBuffer(0, graph.buffer(kCameraBuffer), VK_SHADER_STAGE_FRAGMENT_BIT)
            .depth(false, true)
            .colorAttachment(graph.image(lr::Viewer::kBackbufferName), viewer.swapchainFormat(),
                             VK_ATTACHMENT_LOAD_OP_CLEAR)
            .depthAttachment(graph.image("smoke_demo_scene_depth"), VK_FORMAT_D32_SFLOAT,
                             VK_ATTACHMENT_LOAD_OP_CLEAR, {.depthStencil = {1.0f, 0}})
            .execute([](lr::PassContext &context) { context.cmd().draw(3); });

        lr::eulerian::SmokeRenderPass smokePass({
            .densityResourceName      = solver.densityResourceName(),
            .temperatureResourceName  = solver.temperatureResourceName(),
            .cameraBufferResourceName = kCameraBuffer,
            .depthResourceName        = "smoke_demo_scene_depth",
            .outputImage              = lr::Viewer::kBackbufferName,
            .outputFormat             = viewer.swapchainFormat(),
            // 2 x 3 x 2 world units matches the grid's 48:72:48 aspect.
            .boundsMin                = {-1.0f, 0.0f, -1.0f},
            .boundsMax                = {1.0f, 3.0f, 1.0f},
            .coolColor                = {0.72f, 0.76f, 0.82f},
            .warmColor                = {1.00f, 0.55f, 0.22f},
            .coolTemperature          = simulation.step.ambientTemperature,
            .warmTemperature          = 0.5f * simulation.source.temperature,
            .densityScale             = 1.35f,
            .sampleCount              = 144,
        });
        smokePass.build(graph);
        // Required even with the GUI disabled: the Viewer treats it as the end of pass setup.
        viewer.addImguiPass();

        float elapsed = 0.0f;
        viewer.ownConnection(viewer.onUpdate([&](float dt, VkExtent2D extent) {
            const float aspect =
                static_cast<float>(extent.width) / static_cast<float>(std::max(extent.height, 1u));
            const lr::CameraGpuData camera = makeCamera(aspect);
            viewer.resources().updateBuffer(kCameraBuffer, &camera, sizeof(camera));

            const float simulationDt = std::clamp(dt, 1.0f / 240.0f, 1.0f / 30.0f);
            elapsed += simulationDt;
            // Drifting the source keeps the plume from settling into a perfectly straight column.
            solver.sourceParams().centre[0] = 0.5f + 0.10f * std::sin(elapsed * 1.4f);
            solver.sourceParams().centre[2] = 0.5f + 0.06f * std::sin(elapsed * 0.9f);
            solver.step(simulationDt);
            solver.endStep();
        }));

        const int frameLimit = requestedFrameCount(argc, argv);
        int       frames     = 0;
        while (viewer.step())
        {
            if (frameLimit > 0 && ++frames >= frameLimit) viewer.requestClose();
        }
        return EXIT_SUCCESS;
    } catch (const std::exception &error)
    {
        std::cerr << "smoke_plume_demo failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
