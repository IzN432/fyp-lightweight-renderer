#pragma once

#include "FrameGraphDefinition.hpp"
#include "PassContext.hpp"
#include "PassBuilder.hpp"
#include "ResourceRegistry.hpp"
#include "compiler/GraphCompiler.hpp"
#include "compiler/VulkanBarrierPlanner.hpp"
#include "core/pipeline/Pipeline.hpp"
#include "core/vulkan/Allocator.hpp"
#include "core/vulkan/CommandBuffer.hpp"
#include "core/vulkan/DescriptorAllocator.hpp"
#include "core/vulkan/VulkanContext.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lr
{

class FrameGraph
{
public:
    FrameGraph(const VulkanContext &ctx, ResourceRegistry &registry);
    ~FrameGraph();

    FrameGraph(const FrameGraph &)            = delete;
    FrameGraph &operator=(const FrameGraph &) = delete;

    // Declare a pass — returns a handle-backed builder for fluent configuration.
    // Builders remain valid when later addPass() calls grow the pass array.
    PassBuilder addPass(std::string name);

    // Type-safe frontend identity. String lookup remains inside this adapter
    // layer until ResourceRegistry is split from the logical graph.
    ImageHandle  image(std::string_view name) { return m_graph.image(name); }
    BufferHandle buffer(std::string_view name) { return m_graph.buffer(name); }

    // Access the shared resource registry.
    ResourceRegistry       &resources() { return m_registry; }
    const ResourceRegistry &resources() const { return m_registry; }

    // Compile the graph — topological sort, resource allocation, pipeline creation.
    // Must be called after all passes are declared and resources registered.
    // Also called after swapchain resize.
    void compile();

    // Execute all passes in sorted order for one frame.
    void execute(CommandBuffer &cmd);

    // Call on swapchain resize — rebuilds transient images and recompiles.
    void resize(VkExtent2D newExtent);

    // Typed handles for passes declared so far.
    std::vector<PassHandle> passHandles() const;

    // Deterministic topology snapshot for diagnostics and before/after comparisons.
    // Vulkan object handles are intentionally omitted.
    std::string debugDump() const;

    // Describes a resource layout that the GPU image should be left in after executeAndWait().
    // Used to transition preprocessing outputs (e.g. GENERAL storage writes) into
    // a layout suitable for the main pipeline (e.g. SHADER_READ_ONLY_OPTIMAL).
    struct FinalLayoutDesc
    {
        std::string   resourceName;
        VkImageLayout layout;
    };

    // Compile and synchronously execute this graph's passes. This is intended for
    // short-lived graphs used for preprocessing or uploads. The call waits for the
    // GPU before returning, so the graph can be destroyed immediately afterwards.
    // finalLayouts: optional list of resources to transition at the very end,
    // after all passes have run. The registry is updated to reflect these layouts.
    void executeAndWait(std::vector<FinalLayoutDesc> finalLayouts = {});

    // Inject the current frame's swapchain image before execute().
    // The resource must have been registered via resources().registerExternalImage().
    // After execute() the image will be in COLOR_ATTACHMENT_OPTIMAL —
    // the caller is responsible for transitioning it to PRESENT_SRC_KHR.
    void setExternalImage(const std::string &name, VkImage image, VkImageView view);

private:
    // Compilation steps
    void sortPasses();
    void allocateResources();
    void buildDescriptorSets();
    void buildPipelines();
    void buildBarriers();

    void createDefaultSampler();
    void destroyCompiledPasses();

private:
    const VulkanContext &m_ctx;
    ResourceRegistry    &m_registry;
    FrameGraphDefinition m_graph;
    DescriptorAllocator  m_descriptorAllocator;

    VkSampler m_defaultSampler = VK_NULL_HANDLE;

    struct ExternalImage
    {
        VkImage     image;
        VkImageView view;
    };
    std::unordered_map<std::string, ExternalImage> m_externalImages;

    std::vector<size_t>         m_sortedIndices; // topological order into the definition's passes
    framegraph::GraphDefinition m_logicalGraph;
    framegraph::ExecutionPlan   m_executionPlan;

    struct CompiledImageBarrier
    {
        VkImageMemoryBarrier2 barrier;
        std::string           resourceName; // used to patch external image handles
    };

    struct CompiledBufferBarrier
    {
        VkBufferMemoryBarrier2 barrier;
        std::string            resourceName;
    };

    // Per-pass GPU objects populated by compile()
    struct CompiledPass
    {
        VkDescriptorSetLayout              descriptorLayout = VK_NULL_HANDLE;
        VkPipelineLayout                   pipelineLayout   = VK_NULL_HANDLE;
        VkDescriptorSet                    descriptorSet    = VK_NULL_HANDLE;
        std::unique_ptr<Pipeline>          pipeline;
        VkExtent2D                         renderingExtent{};
        std::vector<CompiledImageBarrier>  imageBarriers;
        std::vector<CompiledBufferBarrier> bufferBarriers;
    };
    std::vector<CompiledPass> m_compiled;

    // Reused per-frame scratch buffers for execute() — cleared (not deallocated) each pass
    // to avoid steady-state heap allocations in the hot path.
    std::vector<VkImageMemoryBarrier2>     m_scratchImageBarriers;
    std::vector<VkBufferMemoryBarrier2>    m_scratchBufferBarriers;
    std::vector<VkRenderingAttachmentInfo> m_scratchColorAttachments;
    VkRenderingAttachmentInfo              m_scratchDepthAttachment{};

    // execute() helpers
    static std::array<float, 4> debugLabelColor(PassType type);
    void                        submitResourceBarriers(CommandBuffer &cmd, const CompiledPass &compiled);
    void                        bindVertexAndIndexBuffers(CommandBuffer &cmd, const PassDesc &pass);
    // Populates m_scratchColorAttachments / m_scratchDepthAttachment from the pass's
    // attachment image uses and returns a VkRenderingInfo referencing them.
    // The returned struct is only valid until the next prepareRenderingInfo() call.
    VkRenderingInfo prepareRenderingInfo(const PassDesc &pass, VkExtent2D extent);
};

} // namespace lr
