#include "FrameGraph.hpp"

#include "FrameGraphTopology.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lr
{

FrameGraph::FrameGraph(const VulkanContext &ctx, ResourceRegistry &registry)
    : m_ctx(ctx), m_registry(registry), m_compiler(ctx, registry)
{}

FrameGraph::~FrameGraph() = default;

PassBuilder FrameGraph::addPass(std::string name)
{
    const PassHandle handle = m_definition.addPass(std::move(name));
    return PassBuilder(m_definition, handle);
}

ImageHandle FrameGraph::importBackbuffer(std::string_view name, VkFormat format)
{
    return m_definition.importBackbuffer(name, format);
}

void FrameGraph::compile()
{
    // Build the replacement completely before releasing the currently usable
    // graph. A compilation failure therefore leaves m_compiled untouched.
    std::unique_ptr<CompiledFrameGraph> replacement = m_compiler.compile(m_definition);
    if (m_compiled)
    {
        m_ctx.waitIdle();
    }
    m_compiled = std::move(replacement);
    ++m_compileCount;

    m_compiledRevision = m_definition.revision();
    m_descriptorGenerations.clear();
    const auto record = [&](const std::string &name) {
        m_descriptorGenerations.emplace_back(name, m_registry.generation(name));
    };
    for (const PassDesc &pass : std::as_const(m_definition).passes())
    {
        for (const ImageUse &use : pass.imageUses)
        {
            if (use.isDescriptor())
            {
                record(m_definition.name(use.image));
            }
        }
        for (const BufferUse &use : pass.bufferUses)
        {
            if (use.isDescriptor())
            {
                record(m_definition.name(use.buffer));
            }
        }
    }
}

bool FrameGraph::needsRecompile() const
{
    if (!m_compiled || m_definition.revision() != m_compiledRevision)
    {
        return true;
    }
    return std::ranges::any_of(m_descriptorGenerations, [&](const auto &entry) {
        return m_registry.generation(entry.first) != entry.second;
    });
}

void FrameGraph::recompileIfNeeded()
{
    // Vertex, index and indirect buffers are looked up by name every frame, so replacing those
    // never needs this; changed passes or replaced descriptor-bound resources do.
    if (m_compiled && needsRecompile())
    {
        compile();
    }
}

void FrameGraph::execute(CommandBuffer &cmd)
{
    if (!m_compiled)
    {
        throw std::logic_error("FrameGraph: execute called before compile");
    }
    recompileIfNeeded();
    m_compiled->execute(cmd);
}

void FrameGraph::execute(CommandBuffer &cmd, const ExternalImageBindings &externalImages)
{
    if (!m_compiled)
    {
        throw std::logic_error("FrameGraph: execute called before compile");
    }
    recompileIfNeeded();
    m_compiled->execute(cmd, externalImages);
}

void FrameGraph::resize(VkExtent2D newExtent)
{
    m_registry.rebuild(newExtent);
    compile();
}

void FrameGraph::executeAndWait(std::vector<FinalLayoutDesc> finalLayouts)
{
    compile();
    m_compiled->executeAndWait(std::move(finalLayouts));
}

void FrameGraph::setExternalImage(const std::string &name, VkImage image, VkImageView view)
{
    if (!m_compiled)
    {
        throw std::logic_error("FrameGraph: external image bound before compile");
    }
    m_compiled->setExternalImage(name, image, view);
}

std::string FrameGraph::debugDump() const
{
    if (m_compiled)
    {
        return m_compiled->debugDump();
    }

    std::vector<size_t> declarationOrder;
    declarationOrder.reserve(m_definition.passes().size());
    for (size_t index = 0; index < m_definition.passes().size(); ++index)
    {
        declarationOrder.push_back(index);
    }
    return framegraph::dumpTopology(m_definition.passes(), m_definition.resources(), declarationOrder);
}

} // namespace lr
