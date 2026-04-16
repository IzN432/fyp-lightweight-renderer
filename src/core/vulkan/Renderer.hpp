#pragma once

#include "CommandBuffer.hpp"
#include "VulkanContext.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace lr
{

class Swapchain;

// Manages per-frame-in-flight resources: command pools, command buffers,
// semaphores, and fences. Owns the record/submit/present loop.
class Renderer
{
public:
    // framesInFlight — how many frames the CPU can prepare ahead of the GPU.
    // 2 is standard (one being rendered, one being recorded).
    Renderer(const VulkanContext &ctx, const Swapchain &swapchain,
             uint32_t framesInFlight = 2);
    ~Renderer();

    Renderer(const Renderer &) = delete;
    Renderer &operator=(const Renderer &) = delete;

    struct FrameResult
    {
        CommandBuffer cmd;
        uint32_t      imageIndex;  // UINT32_MAX if swapchain is out-of-date
    };

    // Wait for the current frame's fence, acquire a swapchain image, begin
    // command buffer recording. Returns the command buffer and image index.
    FrameResult beginFrame(Swapchain &swapchain);

    // End recording, submit, present. Call only if beginFrame returned a valid imageIndex.
    // Returns false if the swapchain needs recreation (resize).
    bool endFrame(Swapchain &swapchain, uint32_t imageIndex);

    // Transition a swapchain image from COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR.
    // Call after all rendering is done, before endFrame().
    static void transitionForPresent(CommandBuffer &cmd, VkImage image,
                                     VkImageLayout oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    uint32_t currentFrame() const { return m_currentFrame; }

private:
    struct FrameData
    {
        VkCommandPool   commandPool    = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer  = VK_NULL_HANDLE;
        VkSemaphore     imageAvailable = VK_NULL_HANDLE;
        VkFence         inFlight       = VK_NULL_HANDLE;
    };

    const VulkanContext &m_ctx;
    std::vector<FrameData>  m_frames;
    std::vector<VkSemaphore> m_renderFinishedPerImage;  // indexed by swapchain image index
    uint32_t m_currentFrame = 0;
};

}  // namespace lr
