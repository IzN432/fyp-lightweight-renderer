#include "features/fluid/ParticleBillboardPass.hpp"

#include "core/Paths.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

namespace
{
struct ParticleBillboardPC
{
    glm::vec4 colorAndRadius;
};
} // namespace

ParticleBillboardPass::ParticleBillboardPass(Config config, ResourceRegistry &resources)
    : m_config(std::move(config)), m_resources(&resources)
{
    if (m_config.maxParticles == 0)
    {
        throw std::invalid_argument("ParticleBillboardPass maxParticles must be greater than zero");
    }

    resources.registerDynamicBuffer(m_config.particleBufferResourceName,
                                    sizeof(glm::vec4) * m_config.maxParticles,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_upload.reserve(m_config.maxParticles);
}

void ParticleBillboardPass::setPositions(std::span<const glm::vec3> positions)
{
    if (positions.size() > m_config.maxParticles)
    {
        throw std::length_error("ParticleBillboardPass particle count exceeds configured capacity");
    }

    m_upload.clear();
    for (const glm::vec3 &position : positions)
    {
        m_upload.emplace_back(position, 1.0f);
    }

    m_particleCount = static_cast<uint32_t>(m_upload.size());
    if (!m_upload.empty())
    {
        m_resources->updateBuffer(m_config.particleBufferResourceName, m_upload.data(),
                                  sizeof(glm::vec4) * m_upload.size());
    }
}

void ParticleBillboardPass::build(FrameGraph &frameGraph)
{
    frameGraph.addPass("fluid_particle_billboards")
        .type(PassType::Geometry)
        .topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .vertShader((paths::shaderDir / "particle_billboard.vert.spv").string())
        .fragShader((paths::shaderDir / "particle_billboard.frag.spv").string())
        .pushConstantSize(sizeof(ParticleBillboardPC), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .uniformBuffer(0, frameGraph.buffer(m_config.cameraBufferResourceName),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(1, frameGraph.buffer(m_config.particleBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledDepth(2, frameGraph.image("gbufferDepth"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .blend(BlendMode::Alpha)
        .depth(true, true, VK_COMPARE_OP_LESS)
        .cull(VK_CULL_MODE_NONE)
        .colorAttachment(frameGraph.image("overlay"), VK_FORMAT_R16G16B16A16_SFLOAT,
                         VK_ATTACHMENT_LOAD_OP_LOAD)
        .depthAttachment(frameGraph.image("overlayDepth"), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_LOAD)
        .execute([this](PassContext &context) {
            if (m_particleCount == 0)
            {
                return;
            }
            const ParticleBillboardPC pc{glm::vec4(m_config.color, m_config.radius)};
            context.cmd().pushConstants(context.pipelineLayout(),
                                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
            context.cmd().draw(6, m_particleCount);
        });
}

} // namespace lr
