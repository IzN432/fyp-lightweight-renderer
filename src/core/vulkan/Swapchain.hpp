#pragma once

#include <vulkan/vulkan.h>

#include <vector>

namespace lr
{

class VulkanContext;
class Window;

class Swapchain
{
public:
    Swapchain(const VulkanContext &ctx, const Window &window);
    ~Swapchain();

    Swapchain(const Swapchain &) = delete;
    Swapchain &operator=(const Swapchain &) = delete;

    // Acquire the next swapchain image. Signals imageAvailable when ready.
    // Returns UINT32_MAX if the swapchain is out-of-date — call recreate() then retry.
    uint32_t acquireNextImage(VkSemaphore imageAvailable);

    // Present the rendered image. Waits on renderFinished before presenting.
    // Returns false if the swapchain is out-of-date or suboptimal — call recreate().
    bool present(uint32_t imageIndex, VkSemaphore renderFinished);

    // Recreate after a resize or out-of-date result.
    void recreate();

    VkImage     getImage(uint32_t index) const { return m_images[index]; }
    VkImageView getImageView(uint32_t index) const { return m_imageViews[index]; }
    VkFormat    getFormat() const { return m_format; }
    VkExtent2D  getExtent() const { return m_extent; }
    uint32_t    imageCount() const { return static_cast<uint32_t>(m_images.size()); }

private:
    void create();
    void destroyImageViews();

    VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats) const;
    VkPresentModeKHR   choosePresentMode(const std::vector<VkPresentModeKHR> &modes) const;
    VkExtent2D         chooseExtent(const VkSurfaceCapabilitiesKHR &caps) const;

    const VulkanContext &m_ctx;
    const Window        &m_window;

    VkSurfaceKHR   m_surface   = VK_NULL_HANDLE;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;

    std::vector<VkImage>     m_images;
    std::vector<VkImageView> m_imageViews;

    VkFormat   m_format = VK_FORMAT_UNDEFINED;
    VkExtent2D m_extent = {};
};

}  // namespace lr
