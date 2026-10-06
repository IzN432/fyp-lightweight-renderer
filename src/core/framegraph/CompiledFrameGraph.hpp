#pragma once

#include "ExternalImageBindings.hpp"
#include "FrameGraphDefinition.hpp"
#include "PassContext.hpp"
#include "ResourceRegistry.hpp"
#include "compiler/GraphCompiler.hpp"
#include "core/pipeline/Pipeline.hpp"
#include "core/vulkan/CommandBuffer.hpp"
#include "core/vulkan/DescriptorAllocator.hpp"
#include "core/vulkan/VulkanContext.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lr
{

class FrameGraphCompiler;

// Immutable graph definition plus the Vulkan objects and execution plan
// produced from it. Construction is reserved for FrameGraphCompiler; callers
// receive this as an opaque executable result.
class CompiledFrameGraph
{
public:
    struct FinalLayoutDesc
    {
        std::string   resourceName;
        VkImageLayout layout;
    };

    ~CompiledFrameGraph();

    CompiledFrameGraph(const CompiledFrameGraph &)            = delete;
    CompiledFrameGraph &operator=(const CompiledFrameGraph &) = delete;
    CompiledFrameGraph(CompiledFrameGraph &&)                 = delete;
    CompiledFrameGraph &operator=(CompiledFrameGraph &&)      = delete;

    void execute(CommandBuffer &cmd);
    void execute(CommandBuffer &cmd, const ExternalImageBindings &externalImages);
    void executeAndWait(std::vector<FinalLayoutDesc> finalLayouts = {});

    void        setExternalImage(const std::string &name, VkImage image, VkImageView view);
    std::string debugDump() const;

private:
    friend class FrameGraphCompiler;

    CompiledFrameGraph(const VulkanContext &ctx, ResourceRegistry &registry, FrameGraphDefinition definition);

    struct ExternalImage
    {
        VkImage     image;
        VkImageView view;
    };

    struct CompiledImageBarrier
    {
        VkImageMemoryBarrier2 barrier;
        std::string           resourceName;
    };

    struct CompiledBufferBarrier
    {
        VkBufferMemoryBarrier2 barrier;
        std::string            resourceName;
    };

    struct CompiledPass
    {
        VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout   = VK_NULL_HANDLE;
        // One set, or one per frame in flight when the pass binds a per-frame (dynamic) buffer;
        // execute() binds the one for the registry's current frame slot.
        std::vector<VkDescriptorSet>       descriptorSets;
        std::unique_ptr<Pipeline>          pipeline;
        VkExtent2D                         renderingExtent{};
        std::vector<CompiledImageBarrier>  imageBarriers;
        std::vector<CompiledBufferBarrier> bufferBarriers;
        // One private multisample image per declared attachment. Rendering resolves these into the
        // ordinary single-sample frame-graph images, so later passes need no MSAA-specific handling.
        std::vector<AllocatedImage>        multisampleAttachments;
    };

    static std::array<float, 4> debugLabelColor(PassType type);
    void submitResourceBarriers(CommandBuffer &cmd, const std::vector<CompiledImageBarrier> &imageBarriers,
                                const std::vector<CompiledBufferBarrier> &bufferBarriers,
                                const ExternalImageBindings              &externalImages);
    const ExternalImageBinding &requireExternalImage(ImageHandle                  image,
                                                     const ExternalImageBindings &externalImages) const;
    void                        bindVertexAndIndexBuffers(CommandBuffer &cmd, const PassDesc &pass);
    VkRenderingInfo             prepareRenderingInfo(size_t passIndex, const PassDesc &pass, VkExtent2D extent,
                                                     const ExternalImageBindings &externalImages);

    const VulkanContext       &m_ctx;
    ResourceRegistry          &m_registry;
    const FrameGraphDefinition m_definition;
    DescriptorAllocator        m_descriptorAllocator;
    VkSampler                  m_defaultSampler = VK_NULL_HANDLE;

    std::unordered_map<std::string, ExternalImage> m_externalImages;
    std::vector<size_t>                            m_sortedIndices;
    framegraph::ExecutionPlan                      m_executionPlan;
    std::vector<CompiledPass>                      m_passes;
    // Logical attachment name -> unresolved multisample image produced for it.
    std::unordered_map<std::string, AllocatedImage *> m_multisampleImages;
    std::vector<CompiledImageBarrier>              m_finalImageBarriers;
    // Each image's layout once a recorded frame has run, written to the registry by execute() (not at
    // compile time: a graph recompiled before this one ever executes must start from the images' real
    // layouts, not from layouts this graph would have left them in).
    std::unordered_map<std::string, VkImageLayout> m_finalImageLayouts;

    std::vector<VkImageMemoryBarrier2>     m_scratchImageBarriers;
    std::vector<VkBufferMemoryBarrier2>    m_scratchBufferBarriers;
    std::vector<VkRenderingAttachmentInfo> m_scratchColorAttachments;
    VkRenderingAttachmentInfo              m_scratchDepthAttachment{};
};

} // namespace lr
