#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/passes/objectpicking/ObjectPickingPass.hpp"

#include <glm/vec3.hpp>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace lr
{

// Blender's selected-object outline: one fullscreen pass that blends an outline over the finished
// image, derived from ObjectPickingPass's ID buffer rather than from a second geometry draw.
//
// The selection arrives as picking IDs (draw index + 1, the same identity ObjectPickingPass writes),
// which the pass packs into a bitset the shader indexes by the ID it reads per pixel. Pixels that
// are not selected but lie within `thickness` pixels of one that is become the outline, so it hugs
// the outside of the visible silhouette and disappears behind nearer geometry — see outline.frag.
//
// Runs after whichever pass produced outputImage: it blends into that attachment rather than
// clearing it, which is also what orders it last (see PassBuilder::blend).
class OutlinePass
{
public:
    struct Config
    {
        VkFormat    outputFormat;
        std::string pickingImage = ObjectPickingPass::imageName;
        std::string outputImage  = "swapchain";
        // Blender's default "Active Object" theme colour.
        glm::vec3 color     = glm::vec3(1.0f, 0.627f, 0.157f);
        float     opacity   = 1.0f;
        float     thickness = 2.0f;
        // Upper bound on the picking IDs the bitset can hold, and so on the scene's draw count.
        // IDs beyond it are treated as unselected rather than overflowing the buffer.
        uint32_t    maxPickingIds       = 65536;
        std::string selectionBufferName = "outlineSelection";
    };

    OutlinePass(Config config, ResourceRegistry &registry);

    void build(FrameGraph &fg);

    // Draws nothing while disabled — the editor hides the outline in states that select vertices
    // rather than objects, as Blender hides it in edit mode.
    void setEnabled(bool enabled) { m_enabled = enabled; }

    // The picking IDs to outline, in any order. IDs are draw indices + 1; 0 is "nothing drawn" and
    // is ignored. Cheap enough to call every frame: it rewrites one small bitset.
    void setSelection(const std::vector<uint32_t> &pickingIds);

    void setStyle(glm::vec3 color, float opacity, float thickness)
    {
        m_config.color     = color;
        m_config.opacity   = opacity;
        m_config.thickness = thickness;
    }

private:
    Config                m_config;
    ResourceRegistry     *m_registry;
    std::vector<uint32_t> m_selectionWords;
    bool                  m_enabled      = true;
    bool                  m_hasSelection = false;
};

} // namespace lr
