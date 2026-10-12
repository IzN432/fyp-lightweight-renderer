#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/framegraph/ResourceRegistry.hpp"

#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace lr
{

// Draws the fluid: one instanced sphere per particle, plus the domain box as an outline. Two
// passes, both writing straight to the swapchain, with no G-buffer, lighting, picking or tone
// mapping in between — so a configured colour reaches the screen unmodified.
//
// This is deliberately separate from the editor's overlay passes. Those exist to draw gizmos over
// a shaded scene, so they carry hover highlighting and occlusion-depth sampling, and the editor's
// final pass tone maps their output. None of that applies to a standalone fluid viewer.
class FluidRenderer
{
public:
    struct Config
    {
        std::string cameraBufferResourceName;
        // Both passes draw straight to the window image, so they need its format.
        VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;
        // Per-particle instance data is re-uploaded each frame, so the buffer is sized once up
        // front for the largest particle count the viewer will show.
        uint32_t maxParticleCount = 262144;
        uint32_t maxLineCount     = 4096;
        // Sphere tessellation. Particles are small on screen, so a coarse sphere is plenty and
        // keeps the per-instance vertex cost down.
        int sphereLatitudeSegments  = 8;
        int sphereLongitudeSegments = 12;
    };

    FluidRenderer(Config config, ResourceRegistry &registry);

    void build(FrameGraph &frameGraph);

    // Particle centres, all drawn at one shared radius and colour.
    void setParticles(const std::vector<glm::vec3> &centers, float radius, const glm::vec3 &color);

    struct Line
    {
        glm::vec3 start{0.0f};
        glm::vec3 end{0.0f};
        glm::vec3 color{1.0f};
    };
    void setLines(const std::vector<Line> &lines);

    // Appends the twelve edges of an axis-aligned box.
    static void appendBoxOutline(std::vector<Line> &lines, const glm::vec3 &min, const glm::vec3 &max,
                                 const glm::vec3 &color);

private:
    struct InstanceData
    {
        glm::vec4 centerAndRadius;
    };

    struct LineVertex
    {
        glm::vec3 position;
        glm::vec3 color;
    };

    struct SphereVertex
    {
        glm::vec3 position;
        glm::vec3 normal;
    };

    void uploadSphereMesh();

    Config            m_config;
    ResourceRegistry *m_registry;

    uint32_t  m_sphereIndexCount = 0;
    uint32_t  m_instanceCount    = 0;
    uint32_t  m_lineVertexCount  = 0;
    glm::vec3 m_particleColor{1.0f};
};

} // namespace lr
