#include "features/fluid/FluidRenderer.hpp"

#include "core/Paths.hpp"
#include "core/app/Viewer.hpp"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace lr
{

namespace
{
} // namespace

FluidRenderer::FluidRenderer(Config config, ResourceRegistry &registry)
    : m_config(std::move(config)), m_registry(&registry)
{
    if (m_config.maxParticleCount == 0)
    {
        throw std::invalid_argument("FluidRenderer maxParticleCount must be greater than zero");
    }

    m_fluidDepth             = m_config.resourceNamePrefix + "fluidDepth";
    m_fluidSphereVertices    = m_config.resourceNamePrefix + "fluidSphereVertices";
    m_fluidSphereIndices     = m_config.resourceNamePrefix + "fluidSphereIndices";
    m_fluidParticleInstances = m_config.resourceNamePrefix + "fluidParticleInstances";
    m_fluidLineVertices      = m_config.resourceNamePrefix + "fluidLineVertices";

    // The sphere mesh never changes, so it is uploaded once here; only the instance buffer is
    // rewritten per frame.
    uploadSphereMesh();

    if (m_config.instanceBufferResourceName.empty())
    {
        m_config.instanceBufferResourceName = m_fluidParticleInstances;
        m_ownsInstanceBuffer                = true;
        // Our own records are the bare centre and radius, with no colour after them.
        m_config.instanceStride = sizeof(InstanceData);
        m_registry->registerDynamicBuffer(m_fluidParticleInstances,
                                          static_cast<VkDeviceSize>(m_config.maxParticleCount) * sizeof(InstanceData),
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    } else if (!m_registry->hasBuffer(m_config.instanceBufferResourceName))
    {
        // Reported here rather than at compile time, where the error would name a missing frame
        // graph resource instead of the component that was meant to register it.
        throw std::invalid_argument("FluidRenderer: instance buffer '" + m_config.instanceBufferResourceName +
                                    "' must be registered before the renderer that reads it");
    }

    m_registry->registerDynamicBuffer(m_fluidLineVertices,
                                      static_cast<VkDeviceSize>(m_config.maxLineCount) * 2 * sizeof(LineVertex),
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
}

void FluidRenderer::uploadSphereMesh()
{
    const int latCount = std::max(m_config.sphereLatitudeSegments, 3);
    const int lonCount = std::max(m_config.sphereLongitudeSegments, 3);

    std::vector<SphereVertex> vertices;
    std::vector<uint32_t>     indices;

    for (int lat = 0; lat <= latCount; ++lat)
    {
        const float theta = static_cast<float>(lat) * glm::pi<float>() / static_cast<float>(latCount);
        for (int lon = 0; lon <= lonCount; ++lon)
        {
            const float     phi = static_cast<float>(lon) * glm::two_pi<float>() / static_cast<float>(lonCount);
            const glm::vec3 position{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
            // Unit sphere centred on the origin, so the normal equals the position.
            vertices.push_back({position, position});
        }
    }

    for (int lat = 0; lat < latCount; ++lat)
    {
        for (int lon = 0; lon < lonCount; ++lon)
        {
            const uint32_t v0 = static_cast<uint32_t>(lat * (lonCount + 1) + lon);
            const uint32_t v1 = v0 + 1;
            const uint32_t v2 = v0 + static_cast<uint32_t>(lonCount + 1);
            const uint32_t v3 = v2 + 1;
            indices.insert(indices.end(), {v0, v2, v1, v1, v2, v3});
        }
    }

    m_sphereIndexCount = static_cast<uint32_t>(indices.size());

    const VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(vertices.size() * sizeof(SphereVertex));
    m_registry->registerDynamicBuffer(m_fluidSphereVertices, vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m_registry->updateBuffer(m_fluidSphereVertices, vertices.data(), vertexBytes);

    const VkDeviceSize indexBytes = static_cast<VkDeviceSize>(indices.size() * sizeof(uint32_t));
    m_registry->registerDynamicBuffer(m_fluidSphereIndices, indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    m_registry->updateBuffer(m_fluidSphereIndices, indices.data(), indexBytes);
}

void FluidRenderer::setParticleCount(uint32_t count, const glm::vec3 &color)
{
    m_particleColor = color;
    m_instanceCount = std::min(count, m_config.maxParticleCount);
}

void FluidRenderer::setParticles(const std::vector<glm::vec3> &centers, float radius, const glm::vec3 &color)
{
    if (!m_ownsInstanceBuffer)
    {
        throw std::logic_error("FluidRenderer: the instance buffer is written by another component; "
                               "use setParticleCount()");
    }
    m_particleColor = color;

    const std::size_t count = std::min<std::size_t>(centers.size(), m_config.maxParticleCount);

    std::vector<InstanceData> instances;
    instances.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        instances.push_back({glm::vec4(centers[i], radius)});
    }

    m_instanceCount = static_cast<uint32_t>(instances.size());
    if (!instances.empty())
    {
        m_registry->updateBuffer(m_config.instanceBufferResourceName, instances.data(),
                                 static_cast<VkDeviceSize>(instances.size() * sizeof(InstanceData)));
    }
}

void FluidRenderer::setLines(const std::vector<Line> &lines)
{
    const std::size_t count = std::min<std::size_t>(lines.size(), m_config.maxLineCount);

    std::vector<LineVertex> vertices;
    vertices.reserve(count * 2);
    for (std::size_t i = 0; i < count; ++i)
    {
        vertices.push_back({lines[i].start, lines[i].color});
        vertices.push_back({lines[i].end, lines[i].color});
    }

    m_lineVertexCount = static_cast<uint32_t>(vertices.size());
    if (!vertices.empty())
    {
        m_registry->updateBuffer(m_fluidLineVertices, vertices.data(),
                                 static_cast<VkDeviceSize>(vertices.size() * sizeof(LineVertex)));
    }
}

void FluidRenderer::appendBoxOutline(std::vector<Line> &lines, const glm::vec3 &min, const glm::vec3 &max,
                                     const glm::vec3 &color)
{
    const glm::vec3 corners[8] = {
        {min.x, min.y, min.z}, {max.x, min.y, min.z}, {max.x, max.y, min.z}, {min.x, max.y, min.z},
        {min.x, min.y, max.z}, {max.x, min.y, max.z}, {max.x, max.y, max.z}, {min.x, max.y, max.z},
    };
    // Four edges around the -Z face, four around +Z, then the four joining the two faces.
    constexpr int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    for (const auto &edge : edges)
    {
        lines.push_back({corners[edge[0]], corners[edge[1]], color});
    }
}

void FluidRenderer::build(FrameGraph &frameGraph)
{
    // ---------------------------------------------------------------------
    // Particles: instanced spheres. Binding 0 is the shared sphere mesh, binding 1 the
    // per-instance centre and radius, which is what makes the whole fluid one draw call rather
    // than one draw per particle.
    // ---------------------------------------------------------------------
    const std::vector<VkVertexInputBindingDescription> particleBindings{
        {.binding = 0, .stride = sizeof(SphereVertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX},
        // The stride comes from the config so an external buffer of wider records -- the GPU
        // solver's, which carry a colour after the centre and radius -- can be read in place.
        {.binding = 1, .stride = m_config.instanceStride, .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE},
    };
    const std::vector<VkVertexInputAttributeDescription> particleAttributes{
        {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(SphereVertex, position)},
        {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(SphereVertex, normal)},
        {.location = 2,
         .binding  = 1,
         .format   = VK_FORMAT_R32G32B32A32_SFLOAT,
         .offset   = offsetof(InstanceData, centerAndRadius)},
    };

    VkClearValue colorClear{};
    // A dark neutral background, so a mid-tone particle colour stays readable against it. The
    // clear value bypasses the fragment shader but is still sRGB-encoded on write, so it is given
    // in linear — these are the linear equivalents of sRGB (0.12, 0.13, 0.15).
    colorClear.color = {{0.0133f, 0.0152f, 0.0194f, 1.0f}};
    VkClearValue depthClear{};
    depthClear.depthStencil = {1.0f, 0};

    frameGraph.addPass(m_config.resourceNamePrefix + "fluid_particles")
        .type(PassType::Geometry)
        .vertexLayout(particleBindings, particleAttributes)
        .vertShader((paths::shaderDir / "fluid_particles.vert.spv").string())
        .fragShader((paths::shaderDir / "fluid_particles.frag.spv").string())
        .pushConstantSize(sizeof(glm::vec4), VK_SHADER_STAGE_FRAGMENT_BIT)
        .uniformBuffer(0, frameGraph.buffer(m_config.cameraBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .vertexBuffer(0, frameGraph.buffer(m_fluidSphereVertices))
        .vertexBuffer(1, frameGraph.buffer(m_config.instanceBufferResourceName))
        .indexBuffer(frameGraph.buffer(m_fluidSphereIndices))
        .depth(true, true)
        .colorAttachment(frameGraph.image(Viewer::kBackbufferName), m_config.swapchainFormat,
                         VK_ATTACHMENT_LOAD_OP_CLEAR, colorClear)
        .depthAttachment(frameGraph.image(m_fluidDepth), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR, depthClear)
        .execute([this](PassContext &context) {
            if (m_instanceCount == 0)
            {
                return;
            }
            context.cmd().pushConstants(context.pipelineLayout(), VK_SHADER_STAGE_FRAGMENT_BIT,
                                        glm::vec4(m_particleColor, 0.0f));
            context.cmd().drawIndexed(m_sphereIndexCount, m_instanceCount);
        });

    // ---------------------------------------------------------------------
    // Domain box. Loads the particle pass's attachments rather than clearing, so it draws over
    // the same frame and shares its depth buffer — particles in front of an edge hide it.
    // ---------------------------------------------------------------------
    const std::vector<VkVertexInputBindingDescription> lineBindings{
        {.binding = 0, .stride = sizeof(LineVertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX},
    };
    const std::vector<VkVertexInputAttributeDescription> lineAttributes{
        {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(LineVertex, position)},
        {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(LineVertex, color)},
    };

    frameGraph.addPass(m_config.resourceNamePrefix + "fluid_lines")
        .type(PassType::Geometry)
        .topology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST)
        .vertexLayout(lineBindings, lineAttributes)
        .vertShader((paths::shaderDir / "fluid_lines.vert.spv").string())
        .fragShader((paths::shaderDir / "fluid_lines.frag.spv").string())
        .uniformBuffer(0, frameGraph.buffer(m_config.cameraBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .vertexBuffer(0, frameGraph.buffer(m_fluidLineVertices))
        .depth(true, true)
        .colorAttachment(frameGraph.image(Viewer::kBackbufferName), m_config.swapchainFormat,
                         VK_ATTACHMENT_LOAD_OP_LOAD)
        .depthAttachment(frameGraph.image(m_fluidDepth), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_LOAD)
        .execute([this](PassContext &context) {
            if (m_lineVertexCount > 0)
            {
                context.cmd().draw(m_lineVertexCount, 1);
            }
        });
}

} // namespace lr
