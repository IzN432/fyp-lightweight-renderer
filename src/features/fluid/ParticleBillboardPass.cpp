#include "features/fluid/ParticleBillboardPass.hpp"

#include "core/Paths.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

ParticleBillboardPass::ParticleBillboardPass(Config config, ResourceRegistry &resources)
    : m_config(std::move(config)), m_resources(&resources)
{
    if (m_config.maxParticles == 0)
    {
        throw std::invalid_argument("ParticleBillboardPass maxParticles must be greater than zero");
    }

    if (m_config.ownsParticleBuffer)
    {
        resources.registerDynamicBuffer(m_config.particleBufferResourceName,
                                        sizeof(ParticleGpuData) * m_config.maxParticles,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    } else if (!resources.hasBuffer(m_config.particleBufferResourceName))
    {
        // Caught here rather than at compile time, where the error would name a missing frame graph
        // resource instead of the component that was meant to register it.
        throw std::invalid_argument("ParticleBillboardPass: particle buffer '" + m_config.particleBufferResourceName +
                                    "' must be registered before the pass that reads it");
    }
}

void ParticleBillboardPass::setParticleCount(uint32_t count)
{
    m_particleCount = particle_billboard::checkedParticleCount(count, m_config.maxParticles);
}

void ParticleBillboardPass::setParticles(std::span<const ParticleGpuData> particles)
{
    if (!m_config.ownsParticleBuffer)
    {
        throw std::logic_error("ParticleBillboardPass: the particle buffer is written by another "
                               "component; use setParticleCount()");
    }
    m_particleCount = particle_billboard::checkedParticleCount(particles.size(), m_config.maxParticles);
    if (!particles.empty())
    {
        m_resources->updateBuffer(m_config.particleBufferResourceName, particles.data(),
                                  sizeof(ParticleGpuData) * particles.size());
    }
}

void ParticleBillboardPass::build(FrameGraph &frameGraph)
{
    frameGraph.addPass("fluid_particle_billboards")
        .type(PassType::Geometry)
        .topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .vertShader((paths::shaderDir / "particle_billboard.vert.spv").string())
        .fragShader((paths::shaderDir / "particle_billboard.frag.spv").string())
        .uniformBuffer(0, frameGraph.buffer(m_config.cameraBufferResourceName),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(1, frameGraph.buffer(m_config.particleBufferResourceName), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledDepth(2, frameGraph.image("gbufferDepth"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .blend(BlendMode::Opaque)
        .depth(true, true, VK_COMPARE_OP_LESS)
        .cull(VK_CULL_MODE_NONE)
        .colorAttachment(frameGraph.image("overlay"), VK_FORMAT_R16G16B16A16_SFLOAT, VK_ATTACHMENT_LOAD_OP_LOAD)
        .depthAttachment(frameGraph.image("overlayDepth"), VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_LOAD)
        .execute([this](PassContext &context) {
            if (m_particleCount == 0)
            {
                return;
            }
            context.cmd().draw(6, m_particleCount);
        });
}

} // namespace lr
