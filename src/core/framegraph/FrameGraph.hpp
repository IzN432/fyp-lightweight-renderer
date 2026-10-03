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
    ImageHandle importBackbuffer(std::string_view name, VkFormat format);

    ImageHandle  image(std::string_view name) { return m_definition.image(name); }
    BufferHandle buffer(std::string_view name) { return m_definition.buffer(name); }

    ResourceRegistry       &resources() { return m_registry; }
    const ResourceRegistry &resources() const { return m_registry; }

    // Builds the graph. Replacing an already compiled graph waits for the device first, since
    // in-flight frames may still use its pipelines and descriptor sets.
    void compile();
    // True when passes changed since compile(), or a resource bound through descriptors was
    // replaced in the registry (its descriptor sets still point at the old handle). execute()
    // recompiles automatically in either case.
    bool needsRecompile() const;
    // How many times compile() has succeeded (for tests and diagnostics).
    uint64_t compileCount() const { return m_compileCount; }

    void execute(CommandBuffer &cmd);
    void execute(CommandBuffer &cmd, const ExternalImageBindings &externalImages);
    void resize(VkExtent2D newExtent);
    void executeAndWait(std::vector<FinalLayoutDesc> finalLayouts = {});
    void setExternalImage(const std::string &name, VkImage image, VkImageView view);

    std::vector<PassHandle> passHandles() const { return m_definition.passHandles(); }
    std::string             debugDump() const;

    FrameGraphDefinition       &definition() { return m_definition; }
    const FrameGraphDefinition &definition() const { return m_definition; }
    const CompiledFrameGraph   *compiled() const { return m_compiled.get(); }

private:
    void recompileIfNeeded();

    const VulkanContext                &m_ctx;
    ResourceRegistry                   &m_registry;
    FrameGraphDefinition                m_definition;
    FrameGraphCompiler                  m_compiler;
    std::unique_ptr<CompiledFrameGraph> m_compiled;

    // State the compiled graph was built from, for needsRecompile().
    uint64_t                                      m_compiledRevision = 0;
    uint64_t                                      m_compileCount     = 0;
    std::vector<std::pair<std::string, uint64_t>> m_descriptorGenerations;
};

} // namespace lr
