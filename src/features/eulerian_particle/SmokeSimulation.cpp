#include "features/eulerian_particle/SmokeSimulation.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>

namespace lr::eulerian
{
namespace
{

// Clamp the simulation timestep so a stalled frame cannot blow up the advection, and a very fast
// one cannot grind the solver down to no visible motion.
constexpr float kMinStepSeconds = 1.0f / 240.0f;
constexpr float kMaxStepSeconds = 1.0f / 30.0f;

} // namespace

glm::vec3 SmokeSimulation::worldSize(const Config &config)
{
    const VkExtent3D &extent = config.solver.extent;
    const float largestAxis =
        static_cast<float>(std::max({extent.width, extent.height, extent.depth}));
    return config.worldWidth / largestAxis *
           glm::vec3(static_cast<float>(extent.width), static_cast<float>(extent.height),
                     static_cast<float>(extent.depth));
}

SmokeSimulation::SmokeSimulation(Config config, ResourceRegistry &resources, const VulkanContext &context)
    : m_config(std::move(config)),
      m_solver(resources, context, m_config.solver),
      m_renderPass([&] {
          // Centred on the origin horizontally and resting on the ground plane, so an emitter near
          // the bottom of the grid sits just above y = 0.
          const glm::vec3 size = worldSize(m_config);
          const glm::vec3 boundsMin{-0.5f * size.x, 0.0f, -0.5f * size.z};
          return SmokeRenderPass::Config{
              .densityResourceName      = m_solver.densityResourceName(),
              .cameraBufferResourceName = m_config.cameraBufferResourceName,
              .depthResourceName        = m_config.depthResourceName,
              .outputImage              = m_config.outputImage,
              .outputFormat             = m_config.outputFormat,
              .boundsMin                = boundsMin,
              .boundsMax                = boundsMin + size,
              .color                    = m_config.color,
              .densityScale             = m_config.densityScale,
              .sampleCount              = m_config.sampleCount,
              .enabled                  = m_config.enabled,
          };
      }())
{
}

void SmokeSimulation::build(FrameGraph &frameGraph) const
{
    m_renderPass.build(frameGraph);
}

void SmokeSimulation::applySourceDrift()
{
    if (m_config.sourceDrift <= 0.0f) return;

    // Two incommensurate frequencies, so the emitter never retraces the same path.
    SmokeSourceParams &source = m_solver.sourceParams();
    source.centre[0] = 0.5f + m_config.sourceDrift * std::sin(m_elapsed * 1.4f);
    source.centre[2] = 0.5f + 0.6f * m_config.sourceDrift * std::sin(m_elapsed * 0.9f);
}

void SmokeSimulation::update(float frameDeltaTime)
{
    if (!m_renderPass.config().enabled) return;
    if (m_paused && !m_singleStepRequested) return;

    const float dt        = std::clamp(frameDeltaTime, kMinStepSeconds, kMaxStepSeconds);
    m_singleStepRequested = false;
    m_elapsed += dt;
    applySourceDrift();

    m_solver.step(dt);
    m_solver.endStep();
}

void SmokeSimulation::reset()
{
    m_solver.reset();
    m_elapsed = 0.0f;
}

void SmokeSimulation::onGUI()
{
    SmokeRenderPass::Config &render = m_renderPass.config();
    SmokeStepParams         &step   = m_solver.stepParams();
    SmokeSourceParams       &source = m_solver.sourceParams();

    if (ImGui::Button(m_paused ? "Play" : "Pause")) m_paused = !m_paused;
    ImGui::SameLine();
    if (ImGui::Button("Step")) m_singleStepRequested = true;
    ImGui::SameLine();
    if (ImGui::Button("Reset")) reset();

    ImGui::Checkbox("Enabled", &render.enabled);

    const VkExtent3D &extent = m_config.solver.extent;
    ImGui::TextDisabled("Grid: %u x %u x %u", extent.width, extent.height, extent.depth);
    ImGui::TextDisabled("Jacobi iterations: %d", m_config.solver.jacobiIterations);
    ImGui::TextDisabled("Grid size and iteration count are fixed at construction.");

    if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderFloat("Buoyancy", &step.buoyancy, 0.0f, 20.0f);
        ImGui::SliderFloat("Gravity", &step.gravity, 0.0f, 20.0f);
        ImGui::SliderFloat("Ambient", &step.ambient, 0.0f, 5.0f);
        ImGui::SliderFloat("Dissipation", &step.densityDamp, 0.0f, 5.0f);
        ImGui::SliderFloat("Velocity Damp", &step.velocityDamp, 0.0f, 5.0f);
        ImGui::SliderFloat("Vorticity", &step.vorticity, 0.0f, 10.0f);
    }

    if (ImGui::CollapsingHeader("Source", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderFloat("Density", &source.density, 0.0f, 50.0f);
        ImGui::SliderFloat("Radius", &source.radius, 0.01f, 0.4f, "%.3f");
        ImGui::SliderFloat3("Velocity", source.velocity, -10.0f, 10.0f);
        // Drift overwrites x and z every step, so editing them by hand only sticks at zero drift.
        ImGui::BeginDisabled(m_config.sourceDrift > 0.0f);
        ImGui::SliderFloat3("Centre", source.centre, 0.0f, 1.0f, "%.3f");
        ImGui::EndDisabled();
        ImGui::SliderFloat("Drift", &m_config.sourceDrift, 0.0f, 0.25f, "%.3f");
    }

    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderFloat("Render Density", &render.densityScale, 0.1f, 20.0f, "%.3f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::SliderInt("Ray-march Samples", &render.sampleCount, 16, 256);
        ImGui::ColorEdit3("Color", &render.color.x);
    }
}

} // namespace lr::eulerian
