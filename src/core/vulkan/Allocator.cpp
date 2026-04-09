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
        throw std::runtime_error("Allocator: failed to create VmaAllocator");

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
        throw std::runtime_error("Allocator: failed to create buffer");

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

AllocatedImage Allocator::createImage(VkExtent3D extent,
                                       VkFormat format,
                                       VkImageUsageFlags imageUsage,
                                       VkImageAspectFlags aspectFlags)
{
    VkImageCreateInfo imageCI{};
    imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = format;
    imageCI.extent = extent;
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = imageUsage;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    AllocatedImage result;
    result.extent = extent;
    result.format = format;

    if (vmaCreateImage(m_allocator, &imageCI, &allocCI,
                       &result.image, &result.allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Allocator: failed to create image");

    // Create image view
    VkImageViewCreateInfo viewCI{};
    viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image = result.image;
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format = format;
    viewCI.subresourceRange.aspectMask = aspectFlags;
    viewCI.subresourceRange.baseMipLevel = 0;
    viewCI.subresourceRange.levelCount = 1;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount = 1;

    // We need the device — get it from VMA's internal info
    VkDevice device;
    vmaGetAllocatorInfo(m_allocator, nullptr);

    // Extract device handle via VMA allocator info
    VmaAllocatorInfo allocInfo;
    vmaGetAllocatorInfo(m_allocator, &allocInfo);
    device = allocInfo.device;

    if (vkCreateImageView(device, &viewCI, nullptr, &result.view) != VK_SUCCESS)
        throw std::runtime_error("Allocator: failed to create image view");

    return result;
}

void Allocator::destroy(AllocatedImage &image)
{
    VmaAllocatorInfo allocInfo;
    vmaGetAllocatorInfo(m_allocator, &allocInfo);

    vkDestroyImageView(allocInfo.device, image.view, nullptr);
    vmaDestroyImage(m_allocator, image.image, image.allocation);

    image.image = VK_NULL_HANDLE;
    image.view = VK_NULL_HANDLE;
    image.allocation = nullptr;
}

}  // namespace lr
