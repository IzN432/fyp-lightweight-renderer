// VMA_IMPLEMENTATION must be defined in exactly one .cpp
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "Allocator.hpp"
#include "VulkanContext.hpp"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace lr
{

Allocator::Allocator(const VulkanContext &ctx)
{
    VmaAllocatorCreateInfo ci{};
    ci.instance = ctx.getInstance();
    ci.physicalDevice = ctx.getPhysicalDevice();
    ci.device = ctx.getDevice();
    ci.vulkanApiVersion = ctx.getApiVersion();
    ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    if (vmaCreateAllocator(&ci, &m_allocator) != VK_SUCCESS)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("Allocator: failed to create VmaAllocator");
    }

    spdlog::info("Allocator: created");
}

Allocator::~Allocator()
{
    vmaDestroyAllocator(m_allocator);
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

AllocatedBuffer Allocator::createBuffer(VkDeviceSize size,
                                         VkBufferUsageFlags bufferUsage,
                                         VmaMemoryUsage memoryUsage)
{
    VkBufferCreateInfo bufferCI{};
    bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCI.size = size;
    bufferCI.usage = bufferUsage;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = memoryUsage;
    allocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;  // persistent mapping for CPU-visible buffers

    AllocatedBuffer result;
    result.size = size;

    if (vmaCreateBuffer(m_allocator, &bufferCI, &allocCI,
                        &result.buffer, &result.allocation, &result.info) != VK_SUCCESS)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("Allocator: failed to create buffer");
    }

    return result;
}

void Allocator::destroy(AllocatedBuffer &buffer)
{
    vmaDestroyBuffer(m_allocator, buffer.buffer, buffer.allocation);
    buffer.buffer = VK_NULL_HANDLE;
    buffer.allocation = nullptr;
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

AllocatedImage Allocator::createImage(const ImageConfig &cfg)
{
    VkImageCreateInfo imageCI{};
    imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.flags = cfg.flags;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = cfg.format;
    imageCI.extent = cfg.extent;
    imageCI.mipLevels = cfg.mipLevels;
    imageCI.arrayLayers = cfg.arrayLayers;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = cfg.usage;
    imageCI.initialLayout = cfg.initialLayout;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    AllocatedImage result;
    result.extent = cfg.extent;
    result.format = cfg.format;
    result.mipLevels = cfg.mipLevels;
    result.arrayLayers = cfg.arrayLayers;

    if (vmaCreateImage(m_allocator, &imageCI, &allocCI,
                       &result.image, &result.allocation, nullptr) != VK_SUCCESS)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("Allocator: failed to create image");
    }

    VmaAllocatorInfo allocInfo;
    vmaGetAllocatorInfo(m_allocator, &allocInfo);
    VkDevice device = allocInfo.device;

    // Full-range view covering all mips and all layers
    VkImageViewCreateInfo viewCI{};
    viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image = result.image;
    viewCI.viewType = cfg.viewType;
    viewCI.format = cfg.format;
    viewCI.subresourceRange.aspectMask = cfg.aspect;
    viewCI.subresourceRange.baseMipLevel = 0;
    viewCI.subresourceRange.levelCount = cfg.mipLevels;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount = cfg.arrayLayers;

    if (vkCreateImageView(device, &viewCI, nullptr, &result.view) != VK_SUCCESS)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("Allocator: failed to create image view");
    }

    return result;
}

AllocatedImage Allocator::createImage(VkExtent3D extent,
                                       VkFormat format,
                                       VkImageUsageFlags usage,
                                       VkImageAspectFlags aspect)
{
    return createImage(ImageConfig{
        .extent = extent,
        .format = format,
        .usage = usage,
        .aspect = aspect,
    });
}

void Allocator::createMipViews(AllocatedImage &image, VkImageAspectFlags aspect)
{
    VmaAllocatorInfo allocInfo;
    vmaGetAllocatorInfo(m_allocator, &allocInfo);
    VkDevice device = allocInfo.device;

    // Infer the view type for per-mip views (cube stays cube, 2D stays 2D)
    VkImageViewType viewType = (image.arrayLayers == 6)
                                   ? VK_IMAGE_VIEW_TYPE_CUBE
                                   : VK_IMAGE_VIEW_TYPE_2D;

    image.mipViews.resize(image.mipLevels);
    for (uint32_t mip = 0; mip < image.mipLevels; ++mip)
    {
        VkImageViewCreateInfo viewCI{};
        viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCI.image = image.image;
        viewCI.viewType = viewType;
        viewCI.format = image.format;
        viewCI.subresourceRange.aspectMask = aspect;
        viewCI.subresourceRange.baseMipLevel = mip;
        viewCI.subresourceRange.levelCount = 1;
        viewCI.subresourceRange.baseArrayLayer = 0;
        viewCI.subresourceRange.layerCount = image.arrayLayers;

        if (vkCreateImageView(device, &viewCI, nullptr, &image.mipViews[mip]) != VK_SUCCESS)
        {
            spdlog::error("Runtime error: throwing std::runtime_error");
            throw std::runtime_error("Allocator: failed to create mip view");
        }
    }
}

void Allocator::destroy(AllocatedImage &image)
{
    VmaAllocatorInfo allocInfo;
    vmaGetAllocatorInfo(m_allocator, &allocInfo);
    VkDevice device = allocInfo.device;

    for (auto view : image.mipViews)
        vkDestroyImageView(device, view, nullptr);
    image.mipViews.clear();

    vkDestroyImageView(device, image.view, nullptr);
    vmaDestroyImage(m_allocator, image.image, image.allocation);

    image.image = VK_NULL_HANDLE;
    image.view = VK_NULL_HANDLE;
    image.allocation = nullptr;
}

}  // namespace lr
