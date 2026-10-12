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

        // Buffer the per-instance centre and radius are read from. Empty means this class owns and
        // uploads it, which is the CPU solver's path. Naming another component's buffer instead --
        // the GPU solver's published particle records -- makes the particles come straight from
        // device memory, with no readback: the records start with a vec4 of centre and radius, so
        // the instance attribute reads them in place and the stride below skips the colour.
        std::string instanceBufferResourceName;
        // Prefixed onto this renderer's own resource names (sphere mesh, line vertices, instance
        // buffer, depth image) so two renderers can coexist in one registry -- which is what lets
        // the viewer keep a CPU-fed and a GPU-fed renderer side by side for comparison.
        std::string resourceNamePrefix;
        // Stride of a record in that buffer, used only when instanceBufferResourceName names an
        // external one. Defaults to sizeof(ParticleGpuData), the layout the GPU solver publishes;
        // when this class owns the buffer it sets the stride to its own narrower record.
        uint32_t instanceStride = 32;
    };

    FluidRenderer(Config config, ResourceRegistry &registry);

    void build(FrameGraph &frameGraph);

    // Particle centres, all drawn at one shared radius and colour. Only valid when this class owns
    // the instance buffer; with an external one use setParticleCount().
    void setParticles(const std::vector<glm::vec3> &centers, float radius, const glm::vec3 &color);

    // How many instances of an externally written buffer to draw, and the colour to shade them.
    void setParticleCount(uint32_t count, const glm::vec3 &color);

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

    uint32_t m_sphereIndexCount   = 0;
    uint32_t m_instanceCount      = 0;
    bool     m_ownsInstanceBuffer = false;

    // This renderer's own registry resources, prefixed per Config::resourceNamePrefix.
    std::string m_fluidDepth;
    std::string m_fluidSphereVertices;
    std::string m_fluidSphereIndices;
    std::string m_fluidParticleInstances;
    std::string m_fluidLineVertices;
    uint32_t    m_lineVertexCount = 0;
    glm::vec3   m_particleColor{1.0f};
};

} // namespace lr
