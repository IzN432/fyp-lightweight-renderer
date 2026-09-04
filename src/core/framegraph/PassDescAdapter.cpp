#include "PassDescAdapter.hpp"

#include <algorithm>
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

lr::framegraph::AccessMode accessMode(lr::BindingAccess access)
{
    switch (access)
    {
        case lr::BindingAccess::Read:
            return lr::framegraph::AccessMode::Read;
        case lr::BindingAccess::Write:
            return lr::framegraph::AccessMode::Write;
        case lr::BindingAccess::ReadWrite:
            return lr::framegraph::AccessMode::ReadWrite;
    }
    return lr::framegraph::AccessMode::Read;
}

struct DescriptorIntent
{
    lr::framegraph::ResourceKind  kind;
    lr::framegraph::ResourceUsage usage;
};

DescriptorIntent descriptorIntent(VkDescriptorType type)
{
    using lr::framegraph::ResourceKind;
    using lr::framegraph::ResourceUsage;

    switch (type)
    {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            return {ResourceKind::Image, ResourceUsage::SampledImage};
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            return {ResourceKind::Image, ResourceUsage::StorageImage};
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            return {ResourceKind::Buffer, ResourceUsage::UniformBuffer};
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            return {ResourceKind::Buffer, ResourceUsage::StorageBuffer};
        default:
            return {ResourceKind::Unknown, ResourceUsage::Unknown};
    }
}

} // namespace

namespace lr::framegraph
{

void resolveTypedHandles(std::span<PassDesc> passes, const ResourceHandleRegistry &resources, uint64_t passOwner)
{
    const auto passName = [&](PassHandle handle) -> const std::string & {
        if (handle.owner != passOwner)
        {
            throw std::invalid_argument("FrameGraph: pass handle belongs to another graph");
        }
        if (handle.index >= passes.size())
        {
            throw std::out_of_range("FrameGraph: invalid pass handle");
        }
        return passes[handle.index].name;
    };

    for (size_t passIndex = 0; passIndex < passes.size(); ++passIndex)
    {
        PassDesc &pass = passes[passIndex];
        if (pass.handle && (pass.handle.owner != passOwner || pass.handle.index != passIndex))
        {
            throw std::invalid_argument("FrameGraph: pass has an invalid frontend handle");
        }

        for (BindingDesc &binding : pass.bindings)
        {
            if (binding.image && binding.buffer)
            {
                throw std::invalid_argument("FrameGraph: binding " + std::to_string(binding.binding) + " in pass '" +
                                            pass.name + "' has both image and buffer handles");
            }

            const std::string *typedName = nullptr;
            if (binding.image)
            {
                if (descriptorIntent(binding.type).kind == ResourceKind::Buffer)
                {
                    throw std::invalid_argument("FrameGraph: image handle used with a buffer descriptor in pass '" +
                                                pass.name + "'");
                }
                typedName = &resources.name(binding.image);
            } else if (binding.buffer)
            {
                if (descriptorIntent(binding.type).kind == ResourceKind::Image)
                {
                    throw std::invalid_argument("FrameGraph: buffer handle used with an image descriptor in pass '" +
                                                pass.name + "'");
                }
                typedName = &resources.name(binding.buffer);
            }

            if (typedName)
            {
                if (!binding.resourceName.empty() && binding.resourceName != *typedName)
                {
                    throw std::invalid_argument("FrameGraph: typed handle and compatibility name disagree in pass '" +
                                                pass.name + "'");
                }
                binding.resourceName = *typedName;
            }
            if (binding.resourceName.empty())
            {
                throw std::invalid_argument("FrameGraph: binding in pass '" + pass.name + "' has no resource");
            }
        }

        for (ResourceDesc &write : pass.writes)
        {
            if (write.image)
            {
                const std::string &typedName = resources.name(write.image);
                if (!write.name.empty() && write.name != typedName)
                {
                    throw std::invalid_argument(
                        "FrameGraph: attachment handle and compatibility name disagree in pass '" + pass.name + "'");
                }
                write.name = typedName;
            }
            if (write.name.empty())
            {
                throw std::invalid_argument("FrameGraph: attachment in pass '" + pass.name + "' has no image");
            }
        }

        for (PassDesc::VertexBufferRef &vertexBuffer : pass.vertexBufferRefs)
        {
            if (vertexBuffer.buffer)
            {
                const std::string &typedName = resources.name(vertexBuffer.buffer);
                if (!vertexBuffer.bufferName.empty() && vertexBuffer.bufferName != typedName)
                {
                    throw std::invalid_argument(
                        "FrameGraph: vertex-buffer handle and compatibility name disagree in pass '" + pass.name + "'");
                }
                vertexBuffer.bufferName = typedName;
            }
            if (vertexBuffer.bufferName.empty())
            {
                throw std::invalid_argument("FrameGraph: vertex binding in pass '" + pass.name + "' has no buffer");
            }
        }

        if (pass.indexBufferHandle)
        {
            const std::string &typedName = resources.name(pass.indexBufferHandle);
            if (!pass.indexBufferName.empty() && pass.indexBufferName != typedName)
            {
                throw std::invalid_argument(
                    "FrameGraph: index-buffer handle and compatibility name disagree in pass '" + pass.name + "'");
            }
            pass.indexBufferName = typedName;
        }

        for (PassHandle dependency : pass.explicitDependencyHandles)
        {
            const std::string &name = passName(dependency);
            if (std::find(pass.explicitDeps.begin(), pass.explicitDeps.end(), name) == pass.explicitDeps.end())
            {
                pass.explicitDeps.push_back(name);
            }
        }
    }
}

GraphDefinition translatePassDescriptions(std::span<const PassDesc> passes)
{
    GraphDefinition     graph;
    std::vector<PassId> passIds;
    passIds.reserve(passes.size());

    for (const PassDesc &pass : passes)
    {
        passIds.push_back(graph.addPass(pass.name, passKind(pass.type)));
    }

    for (size_t passIndex = 0; passIndex < passes.size(); ++passIndex)
    {
        const PassDesc &source      = passes[passIndex];
        const PassId    destination = passIds[passIndex];

        for (const BindingDesc &binding : source.bindings)
        {
            const DescriptorIntent intent   = descriptorIntent(binding.type);
            const ResourceId       resource = graph.addResource(binding.resourceName, intent.kind);
            graph.addAccess(destination, {
                                             .resource = resource,
                                             .mode     = accessMode(binding.access),
                                             .usage    = intent.usage,
                                         });
        }

        for (const ResourceDesc &write : source.writes)
        {
            const ResourceId resource = graph.addResource(write.name, ResourceKind::Image);
            graph.addAccess(destination, {
                                             .resource = resource,
                                             .mode     = AccessMode::Write,
                                             .usage    = ResourceUsage::ColorOrDepthAttachment,
                                         });
        }

        for (const PassDesc::VertexBufferRef &vertexBuffer : source.vertexBufferRefs)
        {
            const ResourceId resource = graph.addResource(vertexBuffer.bufferName, ResourceKind::Buffer);
            graph.addAccess(destination, {
                                             .resource = resource,
                                             .mode     = AccessMode::Read,
                                             .usage    = ResourceUsage::VertexBuffer,
                                         });
        }

        if (!source.indexBufferName.empty())
        {
            const ResourceId resource = graph.addResource(source.indexBufferName, ResourceKind::Buffer);
            graph.addAccess(destination, {
                                             .resource = resource,
                                             .mode     = AccessMode::Read,
                                             .usage    = ResourceUsage::IndexBuffer,
                                         });
        }

        for (const std::string &dependencyName : source.explicitDeps)
        {
            const PassId dependency = graph.findPass(dependencyName);
            if (!dependency)
            {
                throw std::runtime_error("FrameGraph: pass '" + source.name + "' declares dependsOn(\"" +
                                         dependencyName + "\") but no such pass exists");
            }
            graph.addDependency(destination, dependency);
        }
    }

    return graph;
}

} // namespace lr::framegraph
