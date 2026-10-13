#include "features/eulerian_particle/SmokeSimulation.hpp"

#include "core/framegraph/ResourceRegistry.hpp"
#include "core/passes/overlaylines/OverlayLine.hpp"

#include <glm/ext/matrix_transform.hpp>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace lr::eulerian
{
namespace
{

// Clamp the simulation timestep so a stalled frame cannot blow up the advection, and a very fast
// one cannot grind the solver down to no visible motion.
constexpr float kMinStepSeconds = 1.0f / 240.0f;
constexpr float kMaxStepSeconds = 1.0f / 30.0f;

// The solver enforces a one-cell boundary, so a shorter axis would have no interior to simulate.
constexpr uint32_t kMinCellsPerAxis = 3;

// Suffix on the solver's name prefix, so the mask sits alongside the fields it will constrain.
constexpr const char *kObstacleMaskSuffix = "_obstacle_mask";

constexpr OverlayLineStyle kBoundsStyle{
    .color           = {0.45f, 0.70f, 0.95f},
    .visibleOpacity  = 0.9f,
    .occludedOpacity = 0.2f,
};

} // namespace

glm::uvec3 SmokeSimulation::clampedGridSize() const
{
    const uint32_t limit = std::max(m_config.maxCellsPerAxis, kMinCellsPerAxis);
    return glm::uvec3(std::clamp(m_config.gridSize.x, kMinCellsPerAxis, limit),
                      std::clamp(m_config.gridSize.y, kMinCellsPerAxis, limit),
                      std::clamp(m_config.gridSize.z, kMinCellsPerAxis, limit));
}

VkExtent3D SmokeSimulation::gridExtent() const
{
    const glm::uvec3 cells = clampedGridSize();
    return VkExtent3D{cells.x, cells.y, cells.z};
}

void SmokeSimulation::registerObstacleMask(ResourceRegistry &resources) const
{
    // One mask cell per allocated grid cell, so the solver's kernels can index the mask and the
    // fields with the same coordinates. Registered unconditionally, so a debug view can bind it
    // whether or not there is any scene geometry to voxelize this run. STORAGE is what the
    // voxelizer and its clear write through; SAMPLED lets a debug view read it. Exported like the
    // solver's own fields, which is what lets CUDA import it and read it as obstacles.
    const uint32_t    limit = std::max(m_config.maxCellsPerAxis, kMinCellsPerAxis);
    const std::string name  = obstacleMaskResourceName();
    if (resources.getImage(name) != nullptr)
        return;
    resources.registerPersistentVolume(name, VK_FORMAT_R32_UINT,
                                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                       VkExtent3D{limit, limit, limit}, true);
}

SmokeSimulation::SmokeSimulation(Config config, ResourceRegistry &resources, const VulkanContext &context)
    : m_config(std::move(config)),
      m_solver(resources, context,
               [&] {
                   // Allocate at the cell limit on every axis and simulate the sub-region the box
                   // implies. The allocation is what fixes the memory cost; the active extent is
                   // then free to change without touching CUDA's imports or the semaphores.
                   const uint32_t      limit = std::max(m_config.maxCellsPerAxis, kMinCellsPerAxis);
                   SmokeSolver::Config solver = m_config.solver;
                   solver.extent              = VkExtent3D{limit, limit, limit};
                   solver.activeExtent        = gridExtent();

                   // The solver imports the mask, so it has to exist before the solver is
                   // constructed — which is why it is registered here rather than in the body
                   // below, where it would come too late. registerObstacleMask() is idempotent, so
                   // the body's call is harmless.
                   registerObstacleMask(resources);
                   solver.obstacleMaskName = obstacleMaskResourceName();
                   return solver;
               }()),
      m_renderPass([&] {
          return SmokeRenderPass::Config{
              .densityResourceName      = m_solver.densityResourceName(),
              .temperatureResourceName  = m_solver.temperatureResourceName(),
              .cameraBufferResourceName = m_config.cameraBufferResourceName,
              .depthResourceName        = m_config.depthResourceName,
              .outputImage              = m_config.outputImage,
              .outputFormat             = m_config.outputFormat,
              // Overwritten by applyBounds() below; set here only because the field has no default
              // that would be right for this box.
              .boundsMin                = m_config.boundsCentre,
              .boundsMax                = m_config.boundsCentre,
              .coolColor                = m_config.coolColor,
              .warmColor                = m_config.warmColor,
              .coolTemperature          = m_config.coolTemperature,
              .warmTemperature          = m_config.warmTemperature,
              .densityScale             = m_config.densityScale,
              .sampleCount              = m_config.sampleCount,
              .enabled                  = m_config.enabled,
          };
      }())
{
    // The solver was built against the clamped grid above, so record that as what is simulated and
    // publish the box it implies. The render pass must never be left mapping a box the solver is
    // not simulating.
    m_simulatedExtent = gridExtent();

    // The mask volume itself was registered by the initializer above, since the solver had to be
    // able to import it; what is left here is handing its name and size to the voxelizer that
    // writes it.
    const uint32_t    limit = std::max(m_config.maxCellsPerAxis, kMinCellsPerAxis);
    const VkExtent3D  allocated{limit, limit, limit};
    const std::string maskName = obstacleMaskResourceName();

    // No index buffer means no scene to replay — the standalone demo — so there is nothing to
    // voxelize and the mask simply stays cleared.
    if (!m_config.obstacleGeometry.indexBufferResourceName.empty())
    {
        m_voxelizePass.emplace(SmokeVoxelizePass::Config{
            .maskResourceName = maskName,
            .geometry         = m_config.obstacleGeometry,
            // Overwritten by applyVoxelizeBounds() below, as the render pass's box is.
            .boundsMin        = m_config.boundsCentre,
            .boundsMax        = m_config.boundsCentre,
            .gridSize         = m_simulatedExtent,
            .allocatedSize    = allocated,
            .enabled          = m_config.voxelizeObstacles,
        });
    }

    applyBounds();
    applySourceParams();
}

std::string SmokeSimulation::obstacleMaskResourceName() const
{
    return m_config.solver.namePrefix + kObstacleMaskSuffix;
}

void SmokeSimulation::setSceneGeometry(SceneDrawList draws)
{
    if (m_voxelizePass) m_voxelizePass->setSceneGeometry(std::move(draws));
}

void SmokeSimulation::setSkinningEnabled(bool enabled)
{
    if (m_voxelizePass) m_voxelizePass->setSkinningEnabled(enabled);
}

void SmokeSimulation::applyVoxelizeBounds()
{
    if (!m_voxelizePass) return;

    // The same box the renderer maps the volume onto, so the mask's cells line up with the density
    // cells exactly. While a resize is pending that is the box the live fields were simulated in,
    // which is also the grid the solver is still stepping — so the two stay consistent through a
    // resize without any special handling here.
    const SmokeRenderPass::Config &render    = m_renderPass.config();
    SmokeVoxelizePass::Config     &voxelize  = m_voxelizePass->config();
    voxelize.boundsMin                       = render.boundsMin;
    voxelize.boundsMax                       = render.boundsMax;
    voxelize.gridSize                        = m_gridDirty && m_simulatedExtent.width != 0 ? m_simulatedExtent
                                                                                           : gridExtent();
    voxelize.enabled                         = m_config.voxelizeObstacles;
}

void SmokeSimulation::build(FrameGraph &frameGraph, const GpuMeshLayout *layout) const
{
    if (m_voxelizePass)
    {
        if (layout == nullptr)
        {
            throw std::invalid_argument(
                "SmokeSimulation::build: obstacle geometry was configured, so the geometry vertex layout is required");
        }
        m_voxelizePass->build(frameGraph, *layout);
    }
    m_renderPass.build(frameGraph);
}

void SmokeSimulation::setGridSize(const glm::uvec3 &cells)
{
    m_config.gridSize = cells;

    if (m_simulatedExtent.width != 0)
    {
        const VkExtent3D now = gridExtent();
        const bool resized   = now.width != m_simulatedExtent.width || now.height != m_simulatedExtent.height ||
                             now.depth != m_simulatedExtent.depth;
        // Compared against the extent actually simulated rather than against the previous edit, so
        // dragging the resolution away and back again leaves nothing pending. The restart itself
        // is deferred to the next play: doing it here would clear the plume on every frame of a
        // drag, and the running simulation is meant to carry on untouched until then.
        m_gridDirty = resized;
        if (resized) m_paused = true;
    }

    applyBounds();
    applySourceParams();
}

void SmokeSimulation::setBoundsCentre(const glm::vec3 &boundsCentre)
{
    // Pure placement: the grid is unchanged, so this never dirties the simulation.
    m_config.boundsCentre = boundsCentre;
    applyBounds();
    applySourceParams();
}

void SmokeSimulation::setSimulationScale(float scale)
{
    // Rescales the box about its centre and, with it, the plume. The grid is untouched, so
    // like a move this costs nothing: the ray-march pass simply maps the volume onto a bigger box.
    m_config.simulationScale = std::max(scale, 1e-4f);
    applyBounds();
    applySourceParams();
}

void SmokeSimulation::applyBounds()
{
    // While a resize is pending the volumes still hold the previous grid's fields, so the box the
    // renderer maps them onto has to be the one they were simulated in. Handing it the new grid
    // would stretch the old plume across it and distort it as the resolution is dragged, when the
    // point of deferring the restart is that the running simulation carries on untouched. The
    // outline still draws the authored box, so the edit stays visible.
    const bool       pending = m_gridDirty && m_simulatedExtent.width != 0;
    const VkExtent3D cells   = pending ? m_simulatedExtent : gridExtent();
    const float      scale   = std::max(m_config.simulationScale, 1e-4f);

    // Centred on the authored position, so growing the grid or the scale expands the box evenly
    // either side rather than pushing one face outwards.
    const glm::vec3 size = scale * glm::vec3(static_cast<float>(cells.width), static_cast<float>(cells.height),
                                             static_cast<float>(cells.depth));

    SmokeRenderPass::Config &render = m_renderPass.config();
    render.boundsMin                = m_config.boundsCentre - 0.5f * size;
    render.boundsMax                = m_config.boundsCentre + 0.5f * size;

    // The simulated region sits in the origin corner of the allocation, so the box maps to this
    // fraction of the texture on each axis.
    const float allocated = static_cast<float>(std::max(m_config.maxCellsPerAxis, kMinCellsPerAxis));
    render.uvScale        = glm::vec3(static_cast<float>(cells.width), static_cast<float>(cells.height),
                                      static_cast<float>(cells.depth)) /
                     allocated;

    // The voxelizer rasterizes into the same box, so it follows every change to it.
    applyVoxelizeBounds();
}

void SmokeSimulation::applySourceParams()
{
    // Normalised against the box the renderer is using, which while a resize is pending is the one
    // the live fields were simulated in rather than the authored one. That keeps the emitter in the
    // same world position as the plume it is feeding.
    const SmokeRenderPass::Config &render = m_renderPass.config();
    const glm::vec3 size = glm::max(render.boundsMax - render.boundsMin, glm::vec3(1e-6f));

    glm::vec3 position = m_config.sourcePosition;
    if (m_config.sourceDrift > 0.0f)
    {
        // Two incommensurate frequencies, so the emitter never retraces the same path. Applied to
        // the authored position rather than stored into it, so turning drift off restores it.
        position.x += m_config.sourceDrift * std::sin(m_elapsed * 1.4f);
        position.z += 0.6f * m_config.sourceDrift * std::sin(m_elapsed * 0.9f);
    }

    const glm::vec3    normalized = (position - render.boundsMin) / size;
    SmokeSourceParams &source     = m_solver.sourceParams();
    source.centre[0]              = normalized.x;
    source.centre[1]              = normalized.y;
    source.centre[2]              = normalized.z;

    // The kernel normalizes the radius against the grid's longest axis so a spherical source stays
    // spherical on a non-cube grid, so convert against the world length of that same axis. Cells
    // are cubic, so that length is simply the cell count times the scale.
    const float longest = std::max({size.x, size.y, size.z});
    source.radius       = m_config.sourceRadius / longest;
}

void SmokeSimulation::update(float frameDeltaTime)
{
    if (!m_renderPass.config().enabled) return;
    if (m_paused && !m_singleStepRequested) return;

    if (m_gridDirty)
    {
        // Retarget the kernels and clear the leftover fields. Deferred to here so a drag across
        // several cell counts costs one restart, on resume, rather than one per frame.
        reset();
    }

    const float dt        = std::clamp(frameDeltaTime, kMinStepSeconds, kMaxStepSeconds);
    m_singleStepRequested = false;
    m_elapsed += dt;
    applySourceParams();

    m_solver.step(dt);
    m_solver.endStep();
}

void SmokeSimulation::reset()
{
    // Clear the flag and record the new grid before publishing, so applyBounds() hands the render
    // pass the authored box rather than the one the discarded fields were simulated in.
    m_gridDirty       = false;
    m_simulatedExtent = gridExtent();
    applyBounds();

    m_solver.setActiveExtent(m_simulatedExtent);
    m_solver.reset();
    m_elapsed = 0.0f;
    applySourceParams();
}

void SmokeSimulation::appendOverlayLines(OverlayLineBuilder &builder) const
{
    if (!m_config.showBounds) return;

    // The authored box, not the one being rendered: while a resize is pending the two differ, and
    // the outline is what shows the edit that the deferred restart has yet to apply.
    const glm::uvec3 cells = clampedGridSize();
    const float      scale = std::max(m_config.simulationScale, 1e-4f);
    const glm::vec3  size =
        scale * glm::vec3(static_cast<float>(cells.x), static_cast<float>(cells.y), static_cast<float>(cells.z));

    // The primitive is a unit cube centred on the origin, so translate-then-scale lands its
    // corners exactly on the box's.
    const glm::mat4 model = glm::scale(glm::translate(glm::mat4(1.0f), m_config.boundsCentre), size);
    builder.addPrimitive(primitives::cube.outline, model, kBoundsStyle);
}

void SmokeSimulation::onGUI()
{
    SmokeRenderPass::Config &render = m_renderPass.config();
    SmokeStepParams         &step   = m_solver.stepParams();

    if (ImGui::Button(m_paused ? "Play" : "Pause")) m_paused = !m_paused;
    ImGui::SameLine();
    if (ImGui::Button("Step")) m_singleStepRequested = true;
    ImGui::SameLine();
    if (ImGui::Button("Reset")) reset();

    ImGui::Checkbox("Enabled", &render.enabled);

    const VkExtent3D cells = gridExtent();
    const float      scale = std::max(m_config.simulationScale, 1e-4f);
    ImGui::TextDisabled("Grid: %u x %u x %u cells", cells.width, cells.height, cells.depth);
    ImGui::TextDisabled("Box: %.2f x %.2f x %.2f", static_cast<float>(cells.width) * scale,
                        static_cast<float>(cells.height) * scale, static_cast<float>(cells.depth) * scale);
    ImGui::TextDisabled("Jacobi iterations: %d", m_config.solver.jacobiIterations);
    if (m_gridDirty)
    {
        // The live fields are still the old grid's, which is why the plume keeps playing unchanged
        // until the restart happens.
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f), "Resized from %u x %u x %u: play or reset to restart",
                           m_simulatedExtent.width, m_simulatedExtent.height, m_simulatedExtent.depth);
    }

    if (ImGui::CollapsingHeader("Bounds", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Collapsing headers do not open an ID scope, so sections that reuse a label -- "Position"
        // here and under Source -- would collide on one ImGui ID. One scope per section fixes that.
        ImGui::PushID("bounds");

        ImGui::Checkbox("Show Bounds", &m_config.showBounds);

        // The grid resolution is the box size, in cells, so this is an integer edit with no
        // rounding anywhere: one step of the slider is exactly one cell. It is the only control
        // here that costs a restart, which is why the other two are free to drag while playing.
        const int        limit   = static_cast<int>(std::max(m_config.maxCellsPerAxis, kMinCellsPerAxis));
        const glm::uvec3 current = clampedGridSize();
        int              grid[3] = {static_cast<int>(current.x), static_cast<int>(current.y),
                                    static_cast<int>(current.z)};
        if (ImGui::SliderInt3("Grid (cells)", grid, static_cast<int>(kMinCellsPerAxis), limit))
        {
            setGridSize(glm::uvec3(static_cast<uint32_t>(grid[0]), static_cast<uint32_t>(grid[1]),
                                   static_cast<uint32_t>(grid[2])));
        }

        // World units per cell: scales the box and the plume together without touching the grid,
        // so it is safe to drag mid-simulation.
        float scaleEdit = m_config.simulationScale;
        if (ImGui::DragFloat("Scale", &scaleEdit, 0.001f, 0.001f, 1.0f, "%.4f"))
            setSimulationScale(scaleEdit);

        // The box's centre, so dragging it moves the plume without the box also appearing to
        // change size as the grid or scale is edited.
        glm::vec3 centre = m_config.boundsCentre;
        if (ImGui::DragFloat3("Position", &centre.x, 0.01f)) setBoundsCentre(centre);

        ImGui::TextDisabled("Grid costs a restart; scale and position are free.");

        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::PushID("simulation");

        // Buoyancy is per degree above ambient, so its useful range is far smaller than the
        // per-unit-density force it replaced.
        ImGui::SliderFloat("Buoyancy", &step.buoyancy, 0.0f, 0.1f, "%.4f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Upward force per degree above ambient (beta in Fedkiw eq. 8).");
        ImGui::SliderFloat("Gravity", &step.gravity, 0.0f, 20.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Downward force per unit density (alpha in Fedkiw eq. 8).");
        ImGui::SliderFloat("Ambient Temp", &step.ambientTemperature, -50.0f, 200.0f, "%.0f");
        ImGui::SliderFloat("Cooling", &step.cooling, 0.0f, 20.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Fraction of the gap to ambient the gas loses per second.\n"
                              "Zero keeps the plume hot, so it never stalls.");
        ImGui::SliderFloat("Dissipation", &step.densityDamp, 0.0f, 5.0f);
        ImGui::SliderFloat("Velocity Damp", &step.velocityDamp, 0.0f, 5.0f);
        ImGui::SliderFloat("Vorticity", &step.vorticity, 0.0f, 10.0f);

        ImGui::Checkbox("Open Top", &step.openTop);
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("Lets smoke and flow leave through the top of the box.\n"
                              "Closed, the box is sealed: fluid the plume displaces can only "
                              "return downward, so the plume\nrecirculates and smoke deflected by "
                              "an obstacle is pushed under it rather than around it.");
        }

        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Source", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::PushID("source");

        SmokeSourceParams &source = m_solver.sourceParams();
        ImGui::SliderFloat("Density", &source.density, 0.0f, 50.0f);
        ImGui::SliderFloat3("Velocity", source.velocity, -100.0f, 100.0f);
        ImGui::SliderFloat("Temperature", &source.temperature, -50.0f, 1000.0f, "%.0f");
        ImGui::SliderFloat("Heating", &source.heating, 0.0f, 40.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("How fast emitter cells reach the temperature above.");

        // World-space, so these survive moving and resizing the box; the normalized values the
        // kernel wants are derived from them every step.
        ImGui::DragFloat3("Position", &m_config.sourcePosition.x, 0.01f);
        ImGui::DragFloat("Radius", &m_config.sourceRadius, 0.005f, 0.01f, 2.0f, "%.3f");
        ImGui::SliderFloat("Drift", &m_config.sourceDrift, 0.0f, 0.5f, "%.3f");
        if (ImGui::Button("Centre On Box"))
        {
            // Low in the box rather than at its centre, so the plume has room to rise.
            const glm::vec3 size    = render.boundsMax - render.boundsMin;
            m_config.sourcePosition = render.boundsMin + glm::vec3(0.5f * size.x, 0.15f * size.y, 0.5f * size.z);
        }

        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderFloat("Render Density", &render.densityScale, 0.1f, 20.0f, "%.3f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::SliderInt("Ray-march Samples", &render.sampleCount, 16, 256);
        ImGui::ColorEdit3("Cool Color", &render.coolColor.x);
        ImGui::ColorEdit3("Warm Color", &render.warmColor.x);
        // The ramp's ends are in the solver's temperature units, so the useful range runs from
        // ambient up towards the emitter's temperature. Dragging the cool end past the warm one is
        // allowed here and resolved by the render pass, which keeps the span non-degenerate.
        ImGui::DragFloat("Cool Temperature", &render.coolTemperature, 1.0f);
        ImGui::DragFloat("Warm Temperature", &render.warmTemperature, 1.0f);
    }
}

} // namespace lr::eulerian
