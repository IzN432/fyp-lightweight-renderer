#include "features/voxel/VoxelizePass.hpp"

#include "core/Paths.hpp"

#include <glm/vec4.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lr
{
namespace
{

// Mirrors the push_constant block in voxelize.vert/.frag. The vertex shader consumes the
// transform and the box; the fragment shader consumes the grid size. Both declare the whole struct
// so their offsets agree, as the geometry pass's shaders do.
struct alignas(16) VoxelizePC
{
    glm::mat4 model;
    glm::vec4 boundsMin;
    glm::vec4 boundsInvSize;
    glm::uvec4 gridSize;
    // The rasterized viewport, which is the allocated volume rather than the simulated grid: see
    // the renderingExtent call in build() for why it cannot track a resize.
    glm::uvec4 viewportSize;
    uint32_t  primitiveIdOffset;
    uint32_t  paletteOffset;
    uint32_t  skinEnabled;
    uint32_t  padding;
};

static_assert(sizeof(VoxelizePC) == 144);

// Mirrors the push_constant block in voxelize_clear.comp.
struct alignas(16) VoxelizeClearPC
{
    glm::uvec4 gridSize;
};

// Matches local_size in voxelize_clear.comp.
constexpr uint32_t kClearGroupSize = 4;

uint32_t groupCount(uint32_t extent)
{
    return (extent + kClearGroupSize - 1) / kClearGroupSize;
}

// The binding the mask occupies in the voxelizer's descriptor set. Bindings 0-10 are the geometry
// pass's camera, textures and skinning buffers; this pass reuses the skinning ones at their
// original numbers so the shared skinning.glslh include keeps working unchanged, and takes the
// next free slot for its own output.
constexpr uint32_t kMaskBinding = 11;

} // namespace

VoxelizePass::VoxelizePass(Config config) : m_config(std::move(config))
{
    if (m_config.maskResourceName.empty())
    {
        throw std::invalid_argument("VoxelizePass: maskResourceName is required");
    }
    if (m_config.geometry.indexBufferResourceName.empty())
    {
        throw std::invalid_argument("VoxelizePass: indexBufferResourceName is required");
    }
}

void VoxelizePass::build(FrameGraph &frameGraph, const GpuMeshLayout &layout) const
{
    const ImageHandle mask = frameGraph.image(m_config.maskResourceName);

    // The XOR accumulates, so every cell must start at zero. The clear covers the whole allocation
    // rather than the simulated sub-region: a later resize would otherwise grow the grid into cells
    // still holding an old frame's parity and read them as solid.
    frameGraph.addPass(m_config.maskResourceName + "_clear")
        .type(PassType::Compute)
        .computeShader((paths::shaderDirectory() / "voxelize_clear.comp.spv").string())
        .pushConstantSize(sizeof(VoxelizeClearPC), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageImageWrite(0, mask, VK_SHADER_STAGE_COMPUTE_BIT)
        .execute([this](PassContext &context) {
            const VkExtent3D allocated = m_config.allocatedSize;
            const VoxelizeClearPC constants{
                .gridSize = glm::uvec4(allocated.width, allocated.height, allocated.depth, 0u),
            };
            context.cmd().pushConstants(context.pipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, constants);
            context.cmd().dispatch(groupCount(allocated.width), groupCount(allocated.height),
                                   groupCount(allocated.depth));
        });

    auto pass = frameGraph.addPass(m_config.maskResourceName + "_voxelize").type(PassType::Geometry).vertexLayout(layout);

    for (const auto &[binding, bufferName] : m_config.geometry.vertexBufferResourceNames)
    {
        pass.vertexBuffer(binding, frameGraph.buffer(bufferName));
    }

    pass.indexBuffer(frameGraph.buffer(m_config.geometry.indexBufferResourceName))
        .vertShader((paths::shaderDirectory() / "voxelize.vert.spv").string())
        .fragShader((paths::shaderDirectory() / "voxelize.frag.spv").string())
        // Every triangle must be rasterized regardless of winding: parity counts surface
        // crossings, and discarding back faces would drop exactly the exit crossing that closes
        // each span, leaving every mesh solid from its front face to the far wall of the box.
        .cull(VK_CULL_MODE_NONE)
        // No depth buffer and no depth test: the fragment shader resolves depth itself by toggling
        // a whole column, and a depth test would keep only the nearest surface per column, which
        // is precisely the information parity needs all of.
        .depth(false, false)
        // There are no attachments to take the viewport from — the output is a storage image — so
        // it is set explicitly, and one pixel is then exactly one grid column.
        //
        // Sized to the *allocation*, not the simulated grid: the frame graph is compiled once, so
        // this cannot follow a runtime resize. The vertex shader scales the box to the live grid
        // instead, which leaves the surplus columns outside the clip volume and rasterizing
        // nothing; the fragment shader bounds-checks them too. The cost of the larger viewport is
        // only in columns that produce no fragments.
        .renderingExtent(ExtentSpec::absolute(std::max(m_config.allocatedSize.width, 1u),
                                              std::max(m_config.allocatedSize.height, 1u)))
        .pushConstantSize(sizeof(VoxelizePC), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(7, frameGraph.buffer(m_config.geometry.skinInfluenceEntriesBufferResourceName),
                           VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(8, frameGraph.buffer(m_config.geometry.skinInfluenceOffsetsBufferResourceName),
                           VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(9, frameGraph.buffer(m_config.geometry.skinPositionIndicesBufferResourceName),
                           VK_SHADER_STAGE_VERTEX_BIT)
        .storageBufferRead(10, frameGraph.buffer(m_config.geometry.skinJointMatricesBufferResourceName),
                           VK_SHADER_STAGE_VERTEX_BIT)
        .storageImageReadWrite(kMaskBinding, mask, VK_SHADER_STAGE_FRAGMENT_BIT)
        .execute([this](PassContext &context) {
            if (!m_config.enabled)
            {
                return;
            }

            const glm::vec3 size = m_config.boundsMax - m_config.boundsMin;
            // A degenerate box has no interior to voxelize into, and its inverse would be
            // infinite; leaving the mask cleared is the right answer for it.
            if (size.x <= 0.0f || size.y <= 0.0f || size.z <= 0.0f)
            {
                return;
            }

            const VkExtent3D grid     = m_config.gridSize;
            const VkExtent3D viewport = m_config.allocatedSize;
            for (size_t i = 0; i < m_config.geometry.draws.size(); ++i)
            {
                if (!m_config.geometry.draws.isLive(i))
                {
                    continue;
                }
                const SceneDraw       draw = m_config.geometry.draws.at(i, m_skinningEnabled);
                const VoxelizePC constants{
                    .model             = draw.model,
                    .boundsMin         = glm::vec4(m_config.boundsMin, 0.0f),
                    .boundsInvSize     = glm::vec4(1.0f / size, 0.0f),
                    .gridSize          = glm::uvec4(grid.width, grid.height, grid.depth, 0u),
                    .viewportSize      = glm::uvec4(viewport.width, viewport.height, viewport.depth, 0u),
                    .primitiveIdOffset = draw.primitiveIdOffset,
                    .paletteOffset     = draw.paletteOffset,
                    .skinEnabled       = draw.skinEnabled,
                    .padding           = 0u,
                };
                context.cmd().pushConstants(context.pipelineLayout(),
                                            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, constants);
                context.cmd().drawIndexed(draw.indexCount, 1, draw.firstIndex, draw.vertexOffset, 0);
            }
        });
}

} // namespace lr
