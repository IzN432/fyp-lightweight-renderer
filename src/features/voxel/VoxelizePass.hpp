#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/passes/geometry/GeometryPass.hpp"

#include <glm/vec3.hpp>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

namespace lr
{

// Rasterizes scene meshes into a solid obstacle mask over a simulation grid, so grid- and
// particle-based solvers can share the same scene-to-voxel conversion.
//
// It replays the same SceneDrawList as GeometryPass — same vertex and index buffers, same model
// matrices, same skinning path — but with the simulation box as an orthographic viewport and a
// fragment shader that XORs into a 3D storage image instead of writing attachments. Because it
// re-rasterizes live geometry every frame, animated and skinned meshes are handled with no extra
// work, and nothing has to be voxelized on the CPU or cached.
//
// The mask is *solid*, not a surface shell: see voxelize.frag for how depth parity fills the
// interior, and why a shell would let the solver advect density through the middle of a closed
// mesh. The cost of parity is that open meshes smear solid cells to the far wall of the box.
//
// Two passes are built: a compute clear (the XOR accumulates, so the mask must start at zero) and
// the rasterization itself. Both are declared against the full allocated volume, so resizing the
// simulated grid needs no rebuild.
class VoxelizePass
{
public:
    struct Config
    {
        // The owner registers this as a persistent R32_UINT volume at allocatedSize. A set cell is
        // solid; consumers decide how that occupancy affects their own state.
        std::string maskResourceName;

        // The scene geometry to voxelize, and where it lives on the GPU. Taken from the same
        // SceneGpu::geometryPassConfig() the G-buffer pass uses, so the two cannot disagree about
        // what the scene contains; setSceneGeometry() refreshes it on a geometry rebuild.
        GeometryPass::Config geometry{};

        // The simulation box in world space. The voxelizer's whole projection: world positions are
        // normalized against this box and scaled to cells, so the box's mapping onto the grid is
        // identical to the one the consuming simulation uses.
        glm::vec3 boundsMin{-1.0f, 0.0f, -1.0f};
        glm::vec3 boundsMax{1.0f, 2.0f, 1.0f};

        // The simulated sub-region of the allocation, in cells. The rasterized viewport and the
        // clear dispatch are both sized from this.
        VkExtent3D gridSize{64, 64, 64};

        // The allocated volume size, which the clear covers in full: cells outside the simulated
        // region must still be zero, because a later resize would otherwise expose last frame's
        // parity as solid geometry.
        VkExtent3D allocatedSize{64, 64, 64};

        bool enabled = true;
    };

    explicit VoxelizePass(Config config);

    void build(FrameGraph &frameGraph, const GpuMeshLayout &layout) const;

    // Mirrors GeometryPass::setSceneGeometry: replaced wholesale whenever SceneGpu re-packs.
    void setSceneGeometry(SceneDrawList draws) { m_config.geometry.draws = std::move(draws); }

    // Mirrors GeometryPass::setSkinningEnabled, so the pose this pass voxelizes is the one the
    // G-buffer draws. Colliding with a pose the user cannot see would be baffling.
    void setSkinningEnabled(bool enabled) { m_skinningEnabled = enabled; }

    Config       &config() { return m_config; }
    const Config &config() const { return m_config; }

private:
    Config m_config;
    bool   m_skinningEnabled = true;
};

} // namespace lr
