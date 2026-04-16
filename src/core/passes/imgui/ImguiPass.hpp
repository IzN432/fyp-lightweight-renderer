#pragma once

#include "core/vulkan/CommandBuffer.hpp"
#include "core/vulkan/VulkanContext.hpp"

#include <vulkan/vulkan.h>

namespace lr
{

class Window;
class Swapchain;

// Wraps the ImGui GLFW+Vulkan backend for use as a Custom FrameGraph pass.
// Owns no descriptor pool — delegates that to ImGui via DescriptorPoolSize.
class ImguiPass
{
public:
    ImguiPass(const VulkanContext &ctx, const Window &window,
              const Swapchain &swapchain, uint32_t framesInFlight = 2);
    ~ImguiPass();

    ImguiPass(const ImguiPass &) = delete;
    ImguiPass &operator=(const ImguiPass &) = delete;

    // Call once per frame before fg.execute() — opens a new ImGui frame.
    // Place all ImGui:: calls between beginFrame() and the execute callback.
    void beginFrame();

    // Record ImGui draw commands into cmd for the given swapchain image.
    // Handles vkCmdBeginRendering/EndRendering internally.
    // targetView must already be in COLOR_ATTACHMENT_OPTIMAL (the FrameGraph
    // barrier from the Custom pass's writes() declaration takes care of this).
    void render(CommandBuffer &cmd, VkImageView targetView, VkExtent2D extent);

private:
    const VulkanContext &m_ctx;
};

}  // namespace lr
