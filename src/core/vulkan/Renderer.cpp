#include "Renderer.hpp"

#include "Swapchain.hpp"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace lr
{

Renderer::Renderer(const VulkanContext &ctx, const Swapchain &swapchain,
                   uint32_t framesInFlight)
    : m_ctx(ctx)
{
    VkDevice device = ctx.getDevice();
    m_frames.resize(framesInFlight);

    VkSemaphoreCreateInfo semCI{};
    semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (auto &frame : m_frames)
    {
        VkCommandPoolCreateInfo poolCI{};
        poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCI.queueFamilyIndex = static_cast<uint32_t>(ctx.getGraphicsQueueFamily());
        poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

        if (vkCreateCommandPool(device, &poolCI, nullptr, &frame.commandPool) != VK_SUCCESS)
        {
            spdlog::error("Runtime error: throwing std::runtime_error");
            throw std::runtime_error("Renderer: failed to create command pool");
        }

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool        = frame.commandPool;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &allocInfo, &frame.commandBuffer) != VK_SUCCESS)
        {
            spdlog::error("Runtime error: throwing std::runtime_error");
            throw std::runtime_error("Renderer: failed to allocate command buffer");
        }

        if (vkCreateSemaphore(device, &semCI, nullptr, &frame.imageAvailable) != VK_SUCCESS)
        {
            spdlog::error("Runtime error: throwing std::runtime_error");
            throw std::runtime_error("Renderer: failed to create semaphore");
        }

        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        if (vkCreateFence(device, &fenceCI, nullptr, &frame.inFlight) != VK_SUCCESS)
        {
            spdlog::error("Runtime error: throwing std::runtime_error");
            throw std::runtime_error("Renderer: failed to create fence");
        }
    }

    // One renderFinished semaphore per swapchain image — indexed by image index so
    // the presentation engine always releases the semaphore before we reuse it.
    m_renderFinishedPerImage.resize(swapchain.imageCount());
    for (auto &sem : m_renderFinishedPerImage)
        if (vkCreateSemaphore(device, &semCI, nullptr, &sem) != VK_SUCCESS)
        {
            spdlog::error("Runtime error: throwing std::runtime_error");
            throw std::runtime_error("Renderer: failed to create renderFinished semaphore");
        }

    spdlog::info("Renderer: {} frame(s) in flight, {} swapchain image(s)",
                 framesInFlight, swapchain.imageCount());
}

Renderer::~Renderer()
{
    VkDevice device = m_ctx.getDevice();
    for (auto &sem : m_renderFinishedPerImage)
        vkDestroySemaphore(device, sem, nullptr);
    for (auto &frame : m_frames)
    {
        vkDestroyFence(device, frame.inFlight, nullptr);
        vkDestroySemaphore(device, frame.imageAvailable, nullptr);
        vkDestroyCommandPool(device, frame.commandPool, nullptr);
    }
}

Renderer::FrameResult Renderer::beginFrame(Swapchain &swapchain)
{
    VkDevice device = m_ctx.getDevice();
    FrameData &frame = m_frames[m_currentFrame];

    // Wait for the previous use of this frame slot to finish
    vkWaitForFences(device, 1, &frame.inFlight, VK_TRUE, UINT64_MAX);

    // Acquire swapchain image
    uint32_t imageIndex = swapchain.acquireNextImage(frame.imageAvailable);
    if (imageIndex == UINT32_MAX)
        return {CommandBuffer(frame.commandBuffer), UINT32_MAX};

    // Only reset the fence once we know we're going to submit work
    vkResetFences(device, 1, &frame.inFlight);

    // Reset pool and begin recording
    vkResetCommandPool(device, frame.commandPool, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(frame.commandBuffer, &beginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("Renderer: failed to begin command buffer");
    }

    return {CommandBuffer(frame.commandBuffer), imageIndex};
}

bool Renderer::endFrame(Swapchain &swapchain, uint32_t imageIndex)
{
    FrameData &frame = m_frames[m_currentFrame];

    if (vkEndCommandBuffer(frame.commandBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Renderer: failed to end command buffer");
    }

    // Submit
    VkCommandBufferSubmitInfo cmdSubmit{};
    cmdSubmit.sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmdSubmit.commandBuffer = frame.commandBuffer;

    VkSemaphoreSubmitInfo waitInfo{};
    waitInfo.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    waitInfo.semaphore = frame.imageAvailable;
    waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSemaphoreSubmitInfo signalInfo{};
    signalInfo.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalInfo.semaphore = m_renderFinishedPerImage[imageIndex];
    signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.waitSemaphoreInfoCount   = 1;
    submitInfo.pWaitSemaphoreInfos      = &waitInfo;
    submitInfo.commandBufferInfoCount   = 1;
    submitInfo.pCommandBufferInfos      = &cmdSubmit;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos    = &signalInfo;

    if (vkQueueSubmit2(m_ctx.getGraphicsQueue(), 1, &submitInfo, frame.inFlight) != VK_SUCCESS)
    {
        throw std::runtime_error("Renderer: failed to submit command buffer");
    }

    // Present
    bool ok = swapchain.present(imageIndex, m_renderFinishedPerImage[imageIndex]);

    m_currentFrame = (m_currentFrame + 1) % static_cast<uint32_t>(m_frames.size());
    return ok;
}

void Renderer::transitionForPresent(CommandBuffer &cmd, VkImage image,
                                     VkImageLayout oldLayout)
{
    VkPipelineStageFlags2 srcStage;
    VkAccessFlags2        srcAccess;

    switch (oldLayout)
    {
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        srcStage  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        srcAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        break;
    case VK_IMAGE_LAYOUT_GENERAL:
        srcStage  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        srcAccess = VK_ACCESS_2_SHADER_WRITE_BIT;
        break;
    default:
        srcStage  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        srcAccess = VK_ACCESS_2_MEMORY_WRITE_BIT;
        break;
    }

    VkImageMemoryBarrier2 barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask  = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_NONE;
    barrier.oldLayout     = oldLayout;
    barrier.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.image         = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd.get(), &dep);
}

}  // namespace lr
