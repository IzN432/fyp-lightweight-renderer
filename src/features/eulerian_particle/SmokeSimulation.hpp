#pragma once

#include "features/eulerian_particle/SmokeRenderPass.hpp"
#include "features/eulerian_particle/SmokeSolver.hpp"
#include "features/eulerian_particle/SmokeVoxelizePass.hpp"

#include <glm/vec3.hpp>
#include <vulkan/vulkan.h>

#include <optional>
#include <string>

namespace lr
{
class ResourceRegistry;
class VulkanContext;
class FrameGraph;
class OverlayLineBuilder;
}

namespace lr::eulerian
{

// Application-facing smoke controller. It owns the CUDA solver, the ray-march pass and every knob
// the two expose, so the app only has to build it, step it once a frame and draw its window.
//
// The box is authored in cells rather than in world units: gridSize is the resolution and the
// box's size at once, and simulationScale converts to world space (worldSize = gridSize *
// simulationScale). So the box is always exactly a whole number of cells, cells are always cubic,
// and the box and the grid cannot drift out of step. The three knobs separate cleanly:
//
//   - gridSize        resolution and box size, in cells. Costs a restart.
//   - simulationScale world units per cell, so how big the whole thing is. Free.
//   - boundsCentre    where it sits in the world. Free.
//
// Only the first changes what is being simulated; the other two rescale and move the box and the
// plume inside it, which the ray-march pass handles by mapping the box onto the volume.
//
// Resizing the grid does not reallocate anything. The volumes are allocated once at
// maxCellsPerAxis on every axis and the solver simulates the origin-corner sub-region of that
// allocation, which the render pass maps the box onto via SmokeRenderPass::Config::uvScale. What a
// resize does cost is the field contents: density from a differently-sized grid is meaningless, so
// a resize marks the simulation dirty and the next play clears and restarts it. Until then the
// live simulation keeps playing in the grid it started in.
class SmokeSimulation
{
public:
    struct Config
    {
        SmokeSolver::Config solver{};

        // World-space size of one grid cell, and so the overall scale of the simulation: the box's
        // world size is its cell counts times this. Scaling it zooms the whole plume, box and all,
        // without changing the resolution or costing a restart.
        float simulationScale = 2.0f / 64.0f;

        // The authored box, in cells. This is the grid resolution and the box size both: world
        // size is gridSize * simulationScale, so the box can only ever be a whole number of cells
        // and the two can never disagree.
        glm::uvec3 gridSize{64, 96, 64};

        // Where the box sits in the world: its centre, so that resizing the grid or the scale
        // grows it evenly about this point instead of away from one corner. The box extends half
        // of gridSize * simulationScale either side; moving it never touches the grid, so it never
        // costs a restart. The default is half the default box's height, so it rests on y = 0.
        glm::vec3 boundsCentre{0.0f, 1.5f, 0.0f};

        // Per-axis cell limit. Also the allocated volume size, so raising it costs memory on every
        // field whether or not the grid is ever grown into it: the seven volumes total 13 floats
        // per cell, i.e. ~50 MiB at 100 cells per axis and ~104 MiB at 128.
        uint32_t maxCellsPerAxis = 100;

        // Render-pass hookups. The defaults match the editor's frame graph.
        std::string cameraBufferResourceName;
        std::string depthResourceName = "gbufferDepth";
        std::string outputImage       = "swapchain";
        VkFormat    outputFormat      = VK_FORMAT_UNDEFINED;

        glm::vec3 color{0.82f, 0.86f, 0.92f};
        float     densityScale = 5.0f;
        int       sampleCount  = 128;
        bool      enabled      = true;

        // Emitter position and radius in world space. Stored in world units and converted to the
        // solver's normalized grid units each step, so moving or resizing the box leaves the
        // emitter where the user put it instead of dragging it along with the grid.
        glm::vec3 sourcePosition{0.0f, 0.3f, 0.0f};
        float     sourceRadius = 0.25f;

        // Horizontal wander applied to the emitter, in world units. Keeps the plume from settling
        // into a perfectly straight column. Zero holds the source still.
        float sourceDrift = 0.0f;

        // Draws the box as overlay lines. The simulation only contributes the segments; the caller
        // feeds them to the overlay-lines pass.
        bool showBounds = true;

        // Scene geometry to voxelize into the obstacle mask, so smoke can collide with it. Taken
        // from the same SceneGpu::geometryPassConfig() the G-buffer pass uses. Leave the index
        // buffer name empty to build no voxelizer at all, which is what the standalone demo does:
        // it has no SceneGpu to draw from.
        GeometryPass::Config obstacleGeometry{};

        // Whether the voxelizer runs. The mask is only consumed by the solver once collisions are
        // wired into the kernels; until then this populates it for inspection.
        bool voxelizeObstacles = true;
    };

    // Registers and imports the solver's volumes; the registry and context must outlive this.
    SmokeSimulation(Config config, ResourceRegistry &resources, const VulkanContext &context);

    // `layout` is the engine's geometry vertex layout, used by the obstacle voxelizer to replay
    // scene meshes. Required whenever Config::obstacleGeometry names an index buffer.
    void build(FrameGraph &frameGraph, const GpuMeshLayout *layout = nullptr) const;

    // Steps the solver and closes the Vulkan handshake, so the density volume is safe to sample by
    // the time this returns. A no-op while disabled or paused. Applies a pending grid resize first,
    // which restarts the simulation.
    void update(float frameDeltaTime);

    // Zeroes every field and restarts the plume at the current grid size.
    void reset();

    // Draws controls only; the caller owns the containing ImGui window.
    void onGUI();

    // Appends the box outline, if "show bounds" is on. Keeps the overlay-lines pass with a single
    // owner: the caller collects segments from every producer and submits them once.
    void appendOverlayLines(OverlayLineBuilder &builder) const;

    // Keeps the voxelizer's draw list in step with the G-buffer's, for the caller's
    // SceneGpu::onGeometryRebuilt hook. A no-op when no voxelizer was built.
    void setSceneGeometry(SceneDrawList draws);
    void setSkinningEnabled(bool enabled);

    // The obstacle mask volume: one R32_UINT cell per grid cell, 1 inside scene geometry. Named
    // for a debug view or for the solver to import once collisions reach the kernels.
    std::string obstacleMaskResourceName() const;

    bool enabled() const { return m_renderPass.config().enabled; }
    bool paused() const { return m_paused; }
    void setPaused(bool paused) { m_paused = paused; }

    const glm::vec3 &boundsMin() const { return m_renderPass.config().boundsMin; }
    const glm::vec3 &boundsMax() const { return m_renderPass.config().boundsMax; }

    // The grid resolution, in cells, clamped to [3, maxCellsPerAxis] per axis. This is the box's
    // size too, in units of simulationScale. Changing it away from what is being simulated marks
    // the simulation dirty and auto-pauses rather than continuing on a mismatched grid; changing
    // it back clears that again.
    void setGridSize(const glm::uvec3 &cells);

    // Where the box's centre sits in the world. Free to change at any time: it moves the box and
    // the plume inside it without touching the grid, so it never costs a restart.
    void setBoundsCentre(const glm::vec3 &boundsCentre);

    // World units per cell, so the overall size of the simulation. Free to change at any time:
    // like the position it rescales the box and the plume without altering the grid.
    void  setSimulationScale(float scale);
    float simulationScale() const { return m_config.simulationScale; }

    // Whether the grid has been resized away from the one the current fields were simulated on.
    // Cleared by the restart that update() performs on the next play.
    bool gridDirty() const { return m_gridDirty; }

    // The authored grid resolution, and what the live fields were actually simulated at. The two
    // differ exactly while gridDirty().
    VkExtent3D gridExtent() const;
    VkExtent3D simulatedExtent() const { return m_simulatedExtent; }

private:
    // The authored cell counts, clamped to [3, maxCellsPerAxis] per axis.
    glm::uvec3 clampedGridSize() const;

    // Publishes the box and the sub-region it maps to into the render pass.
    void applyBounds();

    // Converts the world-space emitter into the solver's normalized grid units.
    void applySourceParams();

    // Publishes the box and simulated extent into the voxelizer, so the mask it rasterizes always
    // covers the same cells the solver is stepping.
    void applyVoxelizeBounds();

    Config          m_config;
    SmokeSolver     m_solver;
    SmokeRenderPass m_renderPass;
    // Absent when Config::obstacleGeometry names no index buffer, i.e. there is no scene to
    // voxelize. The mask volume is registered either way, so a debug view can always bind it.
    std::optional<SmokeVoxelizePass> m_voxelizePass;

    // The grid the fields currently in the volumes were simulated on. Compared against the grid
    // the authored box implies to decide whether a restart is owed, so dragging the box away and
    // back again leaves nothing pending.
    VkExtent3D      m_simulatedExtent{0, 0, 0};

    bool            m_paused              = false;
    bool            m_singleStepRequested = false;
    bool            m_gridDirty           = false;
    float           m_elapsed             = 0.0f;
};

} // namespace lr::eulerian
