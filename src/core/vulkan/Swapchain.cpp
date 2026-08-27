#include "Swapchain.hpp"

#include "VulkanContext.hpp"
#include "VkResultUtils.hpp"
#include "core/window/Window.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace lr
{

Swapchain::Swapchain(const VulkanContext &ctx, const Window &window) : m_ctx(ctx), m_window(window)
{
    checkVk(glfwCreateWindowSurface(ctx.getInstance(), window.getHandle(), nullptr, &m_surface),
            "Swapchain: glfwCreateWindowSurface");

    try
    {
        create();
    } catch (...)
    {
        vkDestroySurfaceKHR(m_ctx.getInstance(), m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
        throw;
    }
}

Swapchain::~Swapchain()
{
    destroyImageViews();
    if (m_swapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(m_ctx.getDevice(), m_swapchain, nullptr);
    }
    if (m_surface != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(m_ctx.getInstance(), m_surface, nullptr);
    }
}

// ---------------------------------------------------------------------------
// Public
// ---------------------------------------------------------------------------

uint32_t Swapchain::acquireNextImage(VkSemaphore imageAvailable)
{
    uint32_t index;
    VkResult result =
        vkAcquireNextImageKHR(m_ctx.getDevice(), m_swapchain, UINT64_MAX, imageAvailable, VK_NULL_HANDLE, &index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        return UINT32_MAX;
    }

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throwVkError(result, "Swapchain: vkAcquireNextImageKHR");
    }

    return index;
}

bool Swapchain::present(uint32_t imageIndex, VkSemaphore renderFinished)
{
    VkPresentInfoKHR info{};
    info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores    = &renderFinished;
    info.swapchainCount     = 1;
    info.pSwapchains        = &m_swapchain;
    info.pImageIndices      = &imageIndex;

    VkResult result = vkQueuePresentKHR(m_ctx.getGraphicsQueue(), &info);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
    {
        return false;
    }

    if (result != VK_SUCCESS)
    {
        throwVkError(result, "Swapchain: vkQueuePresentKHR");
    }

    return true;
}

void Swapchain::recreate()
{
    // Handle minimised window — wait until it has a non-zero size
    int w = 0, h = 0;
    while (w == 0 || h == 0)
    {
        glfwGetFramebufferSize(m_window.getHandle(), &w, &h);
        glfwWaitEvents();
    }

    m_ctx.waitIdle();
    create();
}

// ---------------------------------------------------------------------------
// Private
// ---------------------------------------------------------------------------

void Swapchain::create()
{
    VkPhysicalDevice physDevice = m_ctx.getPhysicalDevice();
    VkDevice         device     = m_ctx.getDevice();

    VkSurfaceCapabilitiesKHR caps;
    checkVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDevice, m_surface, &caps),
            "Swapchain: vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    uint32_t formatCount = 0;
    checkVk(vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, m_surface, &formatCount, nullptr),
            "Swapchain: vkGetPhysicalDeviceSurfaceFormatsKHR(count)");
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    checkVk(vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, m_surface, &formatCount, formats.data()),
            "Swapchain: vkGetPhysicalDeviceSurfaceFormatsKHR(data)");
    formats.resize(formatCount);

    uint32_t modeCount = 0;
    checkVk(vkGetPhysicalDeviceSurfacePresentModesKHR(physDevice, m_surface, &modeCount, nullptr),
            "Swapchain: vkGetPhysicalDeviceSurfacePresentModesKHR(count)");
    std::vector<VkPresentModeKHR> modes(modeCount);
    checkVk(vkGetPhysicalDeviceSurfacePresentModesKHR(physDevice, m_surface, &modeCount, modes.data()),
            "Swapchain: vkGetPhysicalDeviceSurfacePresentModesKHR(data)");
    modes.resize(modeCount);

    if (formats.empty() || modes.empty())
    {
        throw std::runtime_error("Swapchain: surface has no formats or present modes");
    }

    VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(formats);
    VkPresentModeKHR   presentMode   = choosePresentMode(modes);
    VkExtent2D         extent        = chooseExtent(caps);

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
    {
        imageCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = m_surface;
    ci.minImageCount    = imageCount;
    ci.imageFormat      = surfaceFormat.format;
    ci.imageColorSpace  = surfaceFormat.colorSpace;
    ci.imageExtent      = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform     = caps.currentTransform;
    ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode      = presentMode;
    ci.clipped          = VK_TRUE;
    ci.oldSwapchain     = m_swapchain; // VK_NULL_HANDLE on first create, old handle on recreate

    VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
    checkVk(vkCreateSwapchainKHR(device, &ci, nullptr, &newSwapchain), "Swapchain: vkCreateSwapchainKHR");

    std::vector<VkImage>     newImages;
    std::vector<VkImageView> newImageViews;
    try
    {
        uint32_t count = 0;
        checkVk(vkGetSwapchainImagesKHR(device, newSwapchain, &count, nullptr),
                "Swapchain: vkGetSwapchainImagesKHR(count)");
        newImages.resize(count);
        checkVk(vkGetSwapchainImagesKHR(device, newSwapchain, &count, newImages.data()),
                "Swapchain: vkGetSwapchainImagesKHR(data)");
        newImages.resize(count);

        newImageViews.resize(newImages.size(), VK_NULL_HANDLE);
        for (size_t i = 0; i < newImages.size(); ++i)
        {
            VkImageViewCreateInfo viewCI{};
            viewCI.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewCI.image            = newImages[i];
            viewCI.viewType         = VK_IMAGE_VIEW_TYPE_2D;
            viewCI.format           = surfaceFormat.format;
            viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            checkVk(vkCreateImageView(device, &viewCI, nullptr, &newImageViews[i]), "Swapchain: vkCreateImageView");
        }
    } catch (...)
    {
        for (VkImageView view : newImageViews)
        {
            if (view != VK_NULL_HANDLE)
            {
                vkDestroyImageView(device, view, nullptr);
            }
        }
        vkDestroySwapchainKHR(device, newSwapchain, nullptr);
        throw;
    }

    destroyImageViews();
    if (m_swapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(device, m_swapchain, nullptr);
    }

    m_swapchain  = newSwapchain;
    m_images     = std::move(newImages);
    m_imageViews = std::move(newImageViews);
    m_format     = surfaceFormat.format;
    m_extent     = extent;

    spdlog::info("Swapchain: {}x{}, {} images, format {}", extent.width, extent.height, m_images.size(),
                 static_cast<int>(m_format));
}

void Swapchain::destroyImageViews()
{
    for (auto view : m_imageViews)
    {
        if (view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(m_ctx.getDevice(), view, nullptr);
        }
    }
    m_imageViews.clear();
}

VkSurfaceFormatKHR Swapchain::chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats) const
{
    for (const auto &f : formats)
    {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return f;
        }
    }

    return formats[0];
}

VkPresentModeKHR Swapchain::choosePresentMode(const std::vector<VkPresentModeKHR> &modes) const
{
    for (auto mode : modes)
    {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return mode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR; // always available
}

VkExtent2D Swapchain::chooseExtent(const VkSurfaceCapabilitiesKHR &caps) const
{
    // If the surface specifies a fixed extent, use it
    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return caps.currentExtent;
    }

    // Otherwise clamp the window's framebuffer size to the surface limits
    int w, h;
    glfwGetFramebufferSize(m_window.getHandle(), &w, &h);

    VkExtent2D extent{static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
    extent.width  = std::clamp(extent.width, caps.minImageExtent.width, caps.maxImageExtent.width);
    extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    return extent;
}

} // namespace lr
