#pragma once

#include "CompiledFrameGraph.hpp"
#include "FrameGraphDefinition.hpp"
#include "PassBuilder.hpp"
#include "compiler/FrameGraphCompiler.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace lr
{

// Compatibility facade for the existing C++ renderer. New frontends can own a
// FrameGraphDefinition directly and ask FrameGraphCompiler for an opaque
// CompiledFrameGraph.
class FrameGraph
{
public:
    using FinalLayoutDesc = CompiledFrameGraph::FinalLayoutDesc;

    FrameGraph(const VulkanContext &ctx, ResourceRegistry &registry);
    ~FrameGraph();

    FrameGraph(const FrameGraph &)            = delete;
    FrameGraph &operator=(const FrameGraph &) = delete;

    PassBuilder addPass(std::string name);

    ImageHandle  image(std::string_view name) { return m_definition.image(name); }
    BufferHandle buffer(std::string_view name) { return m_definition.buffer(name); }

    ResourceRegistry       &resources() { return m_registry; }
    const ResourceRegistry &resources() const { return m_registry; }

    void compile();
    void execute(CommandBuffer &cmd);
    void resize(VkExtent2D newExtent);
    void executeAndWait(std::vector<FinalLayoutDesc> finalLayouts = {});
    void setExternalImage(const std::string &name, VkImage image, VkImageView view);

    std::vector<PassHandle> passHandles() const { return m_definition.passHandles(); }
    std::string             debugDump() const;

    FrameGraphDefinition       &definition() { return m_definition; }
    const FrameGraphDefinition &definition() const { return m_definition; }
    const CompiledFrameGraph   *compiled() const { return m_compiled.get(); }

private:
    ResourceRegistry                   &m_registry;
    FrameGraphDefinition                m_definition;
    FrameGraphCompiler                  m_compiler;
    std::unique_ptr<CompiledFrameGraph> m_compiled;
};

} // namespace lr
