#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace lr
{

class VulkanContext;

struct AllocatedBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo info{};
    VkDeviceSize size = 0;
};

struct AllocatedImage
{
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VkExtent3D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
};

class Allocator
{
public:
    explicit Allocator(const VulkanContext &ctx);
    ~Allocator();

    Allocator(const Allocator &) = delete;
    Allocator &operator=(const Allocator &) = delete;
    Allocator(Allocator &&) = delete;
    Allocator &operator=(Allocator &&) = delete;

    // Buffers
    AllocatedBuffer createBuffer(VkDeviceSize size,
                                  VkBufferUsageFlags bufferUsage,
                                  VmaMemoryUsage memoryUsage);
    void destroy(AllocatedBuffer &buffer);

    // Images
    AllocatedImage createImage(VkExtent3D extent,
                                VkFormat format,
                                VkImageUsageFlags imageUsage,
                                VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT);
    void destroy(AllocatedImage &image);

    VmaAllocator getHandle() const { return m_allocator; }

private:
    VmaAllocator m_allocator = nullptr;
};

}  // namespace lr
