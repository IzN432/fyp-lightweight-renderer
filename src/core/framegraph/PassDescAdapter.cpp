#include "PassDescAdapter.hpp"

#include <stdexcept>

namespace
{

lr::framegraph::PassKind passKind(lr::PassType type)
{
    switch (type)
    {
        case lr::PassType::Geometry:
        case lr::PassType::Fullscreen:
            return lr::framegraph::PassKind::Graphics;
        case lr::PassType::Compute:
            return lr::framegraph::PassKind::Compute;
        case lr::PassType::Custom:
            return lr::framegraph::PassKind::External;
    }
    return lr::framegraph::PassKind::External;
}

lr::framegraph::AccessMode accessMode(lr::AccessMode access)
{
    switch (access)
    {
        case lr::AccessMode::Read:
            return lr::framegraph::AccessMode::Read;
        case lr::AccessMode::Write:
            return lr::framegraph::AccessMode::Write;
        case lr::AccessMode::ReadWrite:
            return lr::framegraph::AccessMode::ReadWrite;
    }
    return lr::framegraph::AccessMode::Read;
}

lr::framegraph::ResourceUsage imageUsage(lr::ImageUsage usage)
{
    using U = lr::framegraph::ResourceUsage;
    switch (usage)
    {
        case lr::ImageUsage::Sampled:
        case lr::ImageUsage::SampledDepth:
        case lr::ImageUsage::SampledMultisample:
        case lr::ImageUsage::SampledArray:
            return U::SampledImage;
        case lr::ImageUsage::Storage:
            return U::StorageImage;
        case lr::ImageUsage::ColorAttachment:
            return U::ColorAttachment;
        case lr::ImageUsage::DepthAttachment:
            return U::DepthAttachment;
    }
    return U::Unknown;
}

lr::framegraph::ResourceUsage bufferUsage(lr::BufferUsage usage)
{
    using U = lr::framegraph::ResourceUsage;
    switch (usage)
    {
        case lr::BufferUsage::Uniform:
            return U::UniformBuffer;
        case lr::BufferUsage::Storage:
            return U::StorageBuffer;
        case lr::BufferUsage::Vertex:
            return U::VertexBuffer;
        case lr::BufferUsage::Index:
            return U::IndexBuffer;
        case lr::BufferUsage::Indirect:
            return U::IndirectBuffer;
    }
    return U::Unknown;
}

} // namespace

namespace lr::framegraph
{

GraphDefinition translatePassDescriptions(std::span<const PassDesc> passes, const ResourceHandleRegistry &resources,
                                          uint64_t passOwner)
{
    GraphDefinition     graph;
    std::vector<PassId> passIds;
    passIds.reserve(passes.size());

    for (size_t index = 0; index < passes.size(); ++index)
    {
        const PassDesc &pass = passes[index];
        if (!pass.handle || pass.handle.owner != passOwner || pass.handle.index != index)
        {
            throw std::invalid_argument("FrameGraph: pass has an invalid frontend handle");
        }
        passIds.push_back(graph.addPass(pass.name, passKind(pass.type)));
        if (pass.runsLast)
        {
            graph.setRunsLast(passIds.back());
        }
    }

    for (size_t passIndex = 0; passIndex < passes.size(); ++passIndex)
    {
        const PassDesc &source      = passes[passIndex];
        const PassId    destination = passIds[passIndex];

        for (const ImageUse &use : source.imageUses)
        {
            const ResourceId resource = graph.addResource(resources.name(use.image), ResourceKind::Image);
            graph.addAccess(destination,
                            {.resource = resource, .mode = accessMode(use.access), .usage = imageUsage(use.usage)});
        }
        for (const BufferUse &use : source.bufferUses)
        {
            const ResourceId resource = graph.addResource(resources.name(use.buffer), ResourceKind::Buffer);
            graph.addAccess(destination,
                            {.resource = resource, .mode = accessMode(use.access), .usage = bufferUsage(use.usage)});
        }
        for (PassHandle dependency : source.explicitDependencies)
        {
            if (dependency.owner != passOwner)
            {
                throw std::invalid_argument("FrameGraph: pass dependency handle belongs to another graph");
            }
            if (dependency.index >= passes.size())
            {
                throw std::out_of_range("FrameGraph: invalid pass dependency handle");
            }
            graph.addDependency(destination, passIds[dependency.index]);
        }
    }
    return graph;
}

} // namespace lr::framegraph
