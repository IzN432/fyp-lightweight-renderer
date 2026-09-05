#include "CompiledFrameGraph.hpp"

#include "FrameGraphTopology.hpp"
#include "core/vulkan/VkFormatUtils.hpp"
#include "core/vulkan/VkResultUtils.hpp"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace lr
{

CompiledFrameGraph::CompiledFrameGraph(const VulkanContext &ctx, ResourceRegistry &registry,
                                       FrameGraphDefinition definition)
    : m_ctx(ctx), m_registry(registry), m_definition(std::move(definition)), m_descriptorAllocator(ctx.getDevice())
{
    VkSamplerCreateInfo info{};
    info.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter    = VK_FILTER_LINEAR;
    info.minFilter    = VK_FILTER_LINEAR;
    info.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.minLod       = 0.0f;
    info.maxLod       = VK_LOD_CLAMP_NONE;
    checkVk(vkCreateSampler(m_ctx.getDevice(), &info, nullptr, &m_defaultSampler), "FrameGraph: vkCreateSampler");
}

CompiledFrameGraph::~CompiledFrameGraph() { vkDestroySampler(m_ctx.getDevice(), m_defaultSampler, nullptr); }

std::array<float, 4> CompiledFrameGraph::debugLabelColor(PassType type)
{
    switch (type)
    {
        case PassType::Geometry:
            return {0.20f, 0.70f, 1.00f, 1.00f};
        case PassType::Fullscreen:
            return {0.20f, 1.00f, 0.50f, 1.00f};
        case PassType::Compute:
            return {1.00f, 0.65f, 0.20f, 1.00f};
        case PassType::Custom:
            return {0.85f, 0.40f, 1.00f, 1.00f};
    }
    return {};
}

void CompiledFrameGraph::submitResourceBarriers(CommandBuffer &cmd, const CompiledPass &compiled)
{
    if (compiled.imageBarriers.empty() && compiled.bufferBarriers.empty())
    {
        return;
    }

    m_scratchImageBarriers.clear();
    for (const CompiledImageBarrier &item : compiled.imageBarriers)
    {
        VkImageMemoryBarrier2 barrier  = item.barrier;
        const auto            external = m_externalImages.find(item.resourceName);
        if (external != m_externalImages.end())
        {
            barrier.image = external->second.image;
        }
        m_scratchImageBarriers.push_back(barrier);
    }

    m_scratchBufferBarriers.clear();
    for (const CompiledBufferBarrier &item : compiled.bufferBarriers)
    {
        m_scratchBufferBarriers.push_back(item.barrier);
    }

    VkDependencyInfo dependency{};
    dependency.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount  = static_cast<uint32_t>(m_scratchImageBarriers.size());
    dependency.pImageMemoryBarriers     = m_scratchImageBarriers.data();
    dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(m_scratchBufferBarriers.size());
    dependency.pBufferMemoryBarriers    = m_scratchBufferBarriers.data();
    vkCmdPipelineBarrier2(cmd.get(), &dependency);
}

VkRenderingInfo CompiledFrameGraph::prepareRenderingInfo(const PassDesc &pass, VkExtent2D extent)
{
    m_scratchColorAttachments.clear();
    m_scratchDepthAttachment = {};
    bool hasDepth            = false;

    for (const ImageUse &use : pass.imageUses)
    {
        if (!use.isAttachment())
        {
            continue;
        }
        const std::string    &name  = m_definition.name(use.image);
        const AllocatedImage *image = m_registry.getImage(name);
        if (!image)
        {
            throw std::runtime_error("CompiledFrameGraph: attachment image '" + name + "' is unavailable");
        }

        VkImageView view     = image->view;
        const auto  external = m_externalImages.find(name);
        if (external != m_externalImages.end())
        {
            view = external->second.view;
        }

        VkRenderingAttachmentInfo attachment{};
        attachment.sType      = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        attachment.imageView  = view;
        attachment.loadOp     = use.loadOp;
        attachment.storeOp    = use.storeOp;
        attachment.clearValue = use.clearValue;

        if (use.usage == ImageUsage::DepthAttachment)
        {
            attachment.imageLayout   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            m_scratchDepthAttachment = attachment;
            hasDepth                 = true;
        } else
        {
            attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            m_scratchColorAttachments.push_back(attachment);
        }
    }

    VkRenderingInfo rendering{};
    rendering.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea           = {{0, 0}, extent};
    rendering.layerCount           = 1;
    rendering.colorAttachmentCount = static_cast<uint32_t>(m_scratchColorAttachments.size());
    rendering.pColorAttachments    = m_scratchColorAttachments.data();
    rendering.pDepthAttachment     = hasDepth ? &m_scratchDepthAttachment : nullptr;
    return rendering;
}

void CompiledFrameGraph::bindVertexAndIndexBuffers(CommandBuffer &cmd, const PassDesc &pass)
{
    for (const BufferUse &use : pass.bufferUses)
    {
        if (use.usage == BufferUsage::Vertex)
        {
            const std::string     &name   = m_definition.name(use.buffer);
            const AllocatedBuffer *buffer = m_registry.getBuffer(name);
            if (!buffer)
            {
                throw std::runtime_error("CompiledFrameGraph: missing vertex buffer '" + name + "'");
            }
            cmd.bindVertexBuffer(use.binding, buffer->buffer);
        }
    }
    for (const BufferUse &use : pass.bufferUses)
    {
        if (use.usage == BufferUsage::Index)
        {
            const std::string     &name   = m_definition.name(use.buffer);
            const AllocatedBuffer *buffer = m_registry.getBuffer(name);
            if (!buffer)
            {
                throw std::runtime_error("CompiledFrameGraph: missing index buffer '" + name + "'");
            }
            cmd.bindIndexBuffer(buffer->buffer);
        }
    }
}

void CompiledFrameGraph::execute(CommandBuffer &cmd)
{
    m_registry.flushUploads();
    const auto passes = m_definition.passes();
    for (size_t index : m_sortedIndices)
    {
        const PassDesc     &pass     = passes[index];
        const CompiledPass &compiled = m_passes[index];
        const bool          labels   = m_ctx.debugNamesEnabled();
        if (labels)
        {
            m_ctx.beginDebugLabel(cmd.get(), pass.name, debugLabelColor(pass.type));
        }

        submitResourceBarriers(cmd, compiled);
        if (compiled.pipeline)
        {
            vkCmdBindPipeline(cmd.get(), compiled.pipeline->bindPoint(), compiled.pipeline->get());
        }
        if (compiled.descriptorSet != VK_NULL_HANDLE && compiled.pipeline)
        {
            vkCmdBindDescriptorSets(cmd.get(), compiled.pipeline->bindPoint(), compiled.pipelineLayout, 0, 1,
                                    &compiled.descriptorSet, 0, nullptr);
        }

        PassContext context(cmd, compiled.pipelineLayout, compiled.renderingExtent, m_definition, m_registry);
        if (pass.type == PassType::Compute || pass.type == PassType::Custom)
        {
            if (pass.executeCallback)
            {
                pass.executeCallback(context);
            }
        } else
        {
            VkRenderingInfo rendering = prepareRenderingInfo(pass, compiled.renderingExtent);
            vkCmdBeginRendering(cmd.get(), &rendering);
            cmd.setViewport(0.0f, 0.0f, static_cast<float>(compiled.renderingExtent.width),
                            static_cast<float>(compiled.renderingExtent.height));
            cmd.setScissor(0, 0, compiled.renderingExtent.width, compiled.renderingExtent.height);
            bindVertexAndIndexBuffers(cmd, pass);
            if (pass.executeCallback)
            {
                pass.executeCallback(context);
            }
            vkCmdEndRendering(cmd.get());
        }

        if (labels)
        {
            m_ctx.endDebugLabel(cmd.get());
        }
    }
}

void CompiledFrameGraph::setExternalImage(const std::string &name, VkImage image, VkImageView view)
{
    m_externalImages[name] = {image, view};
}

void CompiledFrameGraph::executeAndWait(std::vector<FinalLayoutDesc> finalLayouts)
{
    const VkDevice          device = m_ctx.getDevice();
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = static_cast<uint32_t>(m_ctx.getGraphicsQueueFamily());
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

    VkCommandPool pool = VK_NULL_HANDLE;
    checkVk(vkCreateCommandPool(device, &poolInfo, nullptr, &pool),
            "CompiledFrameGraph::executeAndWait: vkCreateCommandPool");

    VkFence                      fence     = VK_NULL_HANDLE;
    bool                         submitted = false;
    std::vector<FinalLayoutDesc> completedLayouts;
    try
    {
        VkCommandBufferAllocateInfo allocation{};
        allocation.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocation.commandPool        = pool;
        allocation.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VkCommandBuffer nativeCommand = VK_NULL_HANDLE;
        checkVk(vkAllocateCommandBuffers(device, &allocation, &nativeCommand),
                "CompiledFrameGraph::executeAndWait: vkAllocateCommandBuffers");

        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        checkVk(vkBeginCommandBuffer(nativeCommand, &begin),
                "CompiledFrameGraph::executeAndWait: vkBeginCommandBuffer");

        CommandBuffer command(nativeCommand);
        execute(command);

        std::vector<VkImageMemoryBarrier2> finalBarriers;
        finalBarriers.reserve(finalLayouts.size());
        for (const FinalLayoutDesc &finalLayout : finalLayouts)
        {
            const AllocatedImage *image = m_registry.getImage(finalLayout.resourceName);
            if (!image)
            {
                continue;
            }
            const VkImageLayout current = m_registry.getImageLayout(finalLayout.resourceName);
            if (current == finalLayout.layout)
            {
                continue;
            }

            VkImageMemoryBarrier2 barrier{};
            barrier.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            barrier.oldLayout     = current;
            barrier.newLayout     = finalLayout.layout;
            barrier.image         = image->image;
            const VkImageAspectFlags aspect =
                isDepthFormat(image->format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange = {aspect, 0, image->mipLevels, 0, image->arrayLayers};
            finalBarriers.push_back(barrier);
            completedLayouts.push_back(finalLayout);
        }

        if (!finalBarriers.empty())
        {
            VkDependencyInfo dependency{};
            dependency.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependency.imageMemoryBarrierCount = static_cast<uint32_t>(finalBarriers.size());
            dependency.pImageMemoryBarriers    = finalBarriers.data();
            vkCmdPipelineBarrier2(nativeCommand, &dependency);
        }
        checkVk(vkEndCommandBuffer(nativeCommand), "CompiledFrameGraph::executeAndWait: vkEndCommandBuffer");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        checkVk(vkCreateFence(device, &fenceInfo, nullptr, &fence),
                "CompiledFrameGraph::executeAndWait: vkCreateFence");

        VkCommandBufferSubmitInfo commandSubmit{};
        commandSubmit.sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        commandSubmit.commandBuffer = nativeCommand;
        VkSubmitInfo2 submit{};
        submit.sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos    = &commandSubmit;
        checkVk(vkQueueSubmit2(m_ctx.getGraphicsQueue(), 1, &submit, fence),
                "CompiledFrameGraph::executeAndWait: vkQueueSubmit2");
        submitted = true;
        checkVk(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
                "CompiledFrameGraph::executeAndWait: vkWaitForFences");
    } catch (...)
    {
        if (submitted)
        {
            (void)vkDeviceWaitIdle(device);
        }
        if (fence != VK_NULL_HANDLE)
        {
            vkDestroyFence(device, fence, nullptr);
        }
        vkDestroyCommandPool(device, pool, nullptr);
        throw;
    }

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);
    for (const FinalLayoutDesc &layout : completedLayouts)
    {
        m_registry.setImageLayout(layout.resourceName, layout.layout);
    }
    spdlog::info("FrameGraph: synchronous execution complete");
}

std::string CompiledFrameGraph::debugDump() const
{
    const auto                                             passes = m_definition.passes();
    std::vector<std::vector<framegraph::BarrierDebugInfo>> barriers(passes.size());
    for (size_t index = 0; index < m_passes.size(); ++index)
    {
        for (const CompiledImageBarrier &item : m_passes[index].imageBarriers)
        {
            barriers[index].push_back({.resourceName = item.resourceName,
                                       .srcStage     = item.barrier.srcStageMask,
                                       .srcAccess    = item.barrier.srcAccessMask,
                                       .dstStage     = item.barrier.dstStageMask,
                                       .dstAccess    = item.barrier.dstAccessMask,
                                       .oldLayout    = item.barrier.oldLayout,
                                       .newLayout    = item.barrier.newLayout});
        }
        for (const CompiledBufferBarrier &item : m_passes[index].bufferBarriers)
        {
            barriers[index].push_back({.resourceName = item.resourceName,
                                       .srcStage     = item.barrier.srcStageMask,
                                       .srcAccess    = item.barrier.srcAccessMask,
                                       .dstStage     = item.barrier.dstStageMask,
                                       .dstAccess    = item.barrier.dstAccessMask});
        }
    }
    return framegraph::dumpTopology(passes, m_definition.resources(), m_sortedIndices, barriers);
}

} // namespace lr
