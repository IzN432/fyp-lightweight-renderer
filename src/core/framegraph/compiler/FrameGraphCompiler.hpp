#pragma once

#include <memory>

namespace lr
{

class CompiledFrameGraph;
class FrameGraphDefinition;
class ResourceRegistry;
class VulkanContext;

// Vulkan backend compiler. It consumes a definition snapshot and returns a
// self-contained executable graph; no compiled state is retained here.
class FrameGraphCompiler
{
public:
    FrameGraphCompiler(const VulkanContext &ctx, ResourceRegistry &registry) : m_ctx(ctx), m_registry(registry) {}

    std::unique_ptr<CompiledFrameGraph> compile(const FrameGraphDefinition &definition) const;

private:
    void allocateResources(CompiledFrameGraph &graph) const;
    void buildDescriptorSets(CompiledFrameGraph &graph) const;
    void buildPipelines(CompiledFrameGraph &graph) const;
    void buildBarriers(CompiledFrameGraph &graph) const;

    const VulkanContext &m_ctx;
    ResourceRegistry    &m_registry;
};

} // namespace lr
