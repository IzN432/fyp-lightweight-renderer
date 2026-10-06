#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/loaders/MaterialStore.hpp"
#include "core/passes/geometry/GeometryPass.hpp"

#include <glm/vec3.hpp>
#include <vulkan/vulkan.h>

#include <functional>
#include <utility>
#include <vector>

namespace lr
{

// Forward-shades glTF BLEND surfaces, which GeometryPass leaves out of the G-buffer. Draws the same
// meshes as GeometryPass, sorted back to front by their bounds' centre, and lights them like PbrPass;
// fragments of non-BLEND materials are discarded. Renders at GeometryPass's 4x MSAA, depth-testing
// (without writing) against its unresolved depth, into its own premultiplied-alpha layer that
// FinalPass composites over the lit scene.
//
// Reads:  everything GeometryPass binds (Config::geometry), lightBufferResourceName, "ibl_irradiance",
//         "ibl_prefiltered", "ibl_brdf_lut" (IBLPass), "ltc1", "ltc2" (PbrPass::uploadResources), and
//         "gbufferDepth" as a read-only depth attachment.
// Writes: outputImage (RGBA16F, premultiplied colour + opacity).
//
// Sorting is per mesh, so overlapping BLEND surfaces within one mesh can still composite out of order.
class TransparentPass
{
public:
    struct Config
    {
        // The scene buffers, textures and materials GeometryPass draws from (see SceneGpu::geometryPassConfig).
        GeometryPass::Config geometry;
        std::string          lightBufferResourceName;
        uint32_t             numLights = 0;
        uint32_t             pfMips    = 8;
        // World-space camera position, read every frame to sort draws.
        std::function<glm::vec3()> eyePosition;
        // Whether a material is BLEND. Read every frame, so a draw whose materials are all opaque is
        // skipped and editing a material's alphaBlend takes effect immediately.
        std::function<bool(MaterialHandle)> isBlendMaterial;
        std::string                         outputImage = "transparent";
    };

    explicit TransparentPass(Config cfg);

    void build(FrameGraph &fg, const GpuMeshLayout &layout) const;

    // The same draw arrays as GeometryPass::setSceneGeometry, plus each draw's mesh (for its bounds and
    // materials). All must be parallel.
    void setSceneGeometry(VertexBufferUploadResult vertices, IndexBufferUploadResult indices,
                          std::vector<const TransformComponent *> transforms, std::vector<SceneObject *> objects,
                          std::vector<SkinDrawInfo> skins, const std::vector<const Mesh *> &meshes);

    void setNumLights(uint32_t numLights) { m_cfg.numLights = numLights; }
    void setSkinningEnabled(bool enabled) { m_skinningEnabled = enabled; }

private:
    struct DrawBounds
    {
        glm::vec3                   localCenter{0.0f};
        std::vector<MaterialHandle> materials; // distinct, from the mesh's face groups
    };

    Config                  m_cfg;
    std::vector<DrawBounds> m_draws;
    bool                    m_skinningEnabled = true;
    // (squared eye distance, draw index), rebuilt every frame.
    mutable std::vector<std::pair<float, size_t>> m_order;
};

} // namespace lr
