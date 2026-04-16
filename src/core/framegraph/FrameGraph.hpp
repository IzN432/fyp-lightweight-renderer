#pragma once

#include "PassBuilder.hpp"
#include "ResourceRegistry.hpp"
#include "core/pipeline/Pipeline.hpp"
#include "core/vulkan/Allocator.hpp"
#include "core/vulkan/CommandBuffer.hpp"
#include "core/vulkan/DescriptorAllocator.hpp"
#include "core/vulkan/VulkanContext.hpp"

#include <vulkan/vulkan.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lr
{

class FrameGraph
{
public:
    FrameGraph(const VulkanContext &ctx, Allocator &allocator, VkExtent2D extent);
    ~FrameGraph();

    FrameGraph(const FrameGraph &) = delete;
    FrameGraph &operator=(const FrameGraph &) = delete;

    // Declare a pass — returns a builder for fluent configuration.
    // The returned reference is valid until compile() is called.
    PassBuilder addPass(const std::string &name);

    // Access the resource registry — register buffers and persistent images here
    // before calling compile().
    ResourceRegistry &resources() { return m_registry; }
    const ResourceRegistry &resources() const { return m_registry; }

    // Compile the graph — topological sort, resource allocation, pipeline creation.
    // Must be called after all passes are declared and resources registered.
    // Also called after swapchain resize.
    void compile();

    // Execute all passes in sorted order for one frame.
    void execute(CommandBuffer &cmd);

    // Call on swapchain resize — rebuilds transient images and recompiles.
    void resize(VkExtent2D newExtent);

    // Returns the names of all currently declared passes, in declaration order.
    // Used by Viewer to build the imgui pass's dependsOn list.
    std::vector<std::string> passNames() const;

    // Compile and execute all currently declared passes exactly once,
    // then discard them. Resources in the registry are preserved.
    // Use for one-shot preprocessing work (IBL generation, etc.)
    // before adding per-frame passes and calling run().
    void executeOnce();

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
    Allocator &m_allocator;
    ResourceRegistry m_registry;
    DescriptorAllocator m_descriptorAllocator;

    VkSampler m_defaultSampler = VK_NULL_HANDLE;

    struct ExternalImage { VkImage image; VkImageView view; };
    std::unordered_map<std::string, ExternalImage>  m_externalImages;

    std::vector<PassDesc> m_passes;         // declared passes (insertion order)
    std::vector<size_t> m_sortedIndices;    // topological order into m_passes

    struct CompiledBarrier
    {
        VkImageMemoryBarrier2 barrier;
        std::string           resourceName;  // used to patch external image handles
    };

    // Per-pass GPU objects populated by compile()
    struct CompiledPass
    {
        VkDescriptorSetLayout             descriptorLayout = VK_NULL_HANDLE;
        VkPipelineLayout                  pipelineLayout   = VK_NULL_HANDLE;
        VkDescriptorSet                   descriptorSet    = VK_NULL_HANDLE;
        std::unique_ptr<Pipeline>         pipeline;
        std::vector<CompiledBarrier>      barriers;
    };
    std::vector<CompiledPass> m_compiled;
};

}  // namespace lr
