#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <vector>

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
    VkImageView view = VK_NULL_HANDLE;           // full range (all mips, all layers)
    std::vector<VkImageView> mipViews;           // per-mip views, created by createMipViews()
    VmaAllocation allocation = nullptr;
    VkExtent3D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
};

// Full image creation config. Use the convenience overload for simple 2D images.
struct ImageConfig
{
    VkExtent3D extent;
    VkFormat format;
    VkImageUsageFlags usage;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkImageCreateFlags flags = 0;                // e.g. VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT
    VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
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
    AllocatedImage createImage(const ImageConfig &cfg);
    // Convenience overload — 2D, 1 mip, 1 layer
    AllocatedImage createImage(VkExtent3D extent,
                                VkFormat format,
                                VkImageUsageFlags usage,
                                VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
    // Creates one VkImageView per mip level (covering all array layers),
    // stored in image.mipViews. Used for compute writes to specific mip levels.
    void createMipViews(AllocatedImage &image,
                         VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
    void destroy(AllocatedImage &image);

    VmaAllocator getHandle() const { return m_allocator; }

private:
    VmaAllocator m_allocator = nullptr;
};

}  // namespace lr
