// VMA_IMPLEMENTATION must be defined in exactly one .cpp
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "Allocator.hpp"
#include "VkResultUtils.hpp"
#include "VulkanContext.hpp"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace lr
{

Allocator::Allocator(const VulkanContext &ctx) : m_ctx(ctx)
{
    VmaAllocatorCreateInfo ci{};
    ci.instance         = ctx.getInstance();
    ci.physicalDevice   = ctx.getPhysicalDevice();
    ci.device           = ctx.getDevice();
    ci.vulkanApiVersion = ctx.getApiVersion();
    ci.flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    checkVk(vmaCreateAllocator(&ci, &m_allocator), "Allocator: vmaCreateAllocator");

    spdlog::info("Allocator: created");
}

Allocator::~Allocator()
{
    if (m_allocator != nullptr)
    {
        vmaDestroyAllocator(m_allocator);
    }
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

AllocatedBuffer Allocator::createBuffer(VkDeviceSize size, VkBufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage)
{
    // An empty payload is a legitimate state (a scene cleared of all geometry, a skinless mesh list),
    // but a zero-sized buffer is not: vkCreateBuffer rejects size 0, and a descriptor written over it
    // would carry range 0. Round up to a byte so the allocation and its descriptors stay valid; the
    // buffer is never read, since whatever draws from it has an empty draw list.
    const VkDeviceSize allocationSize = (size == 0) ? 1 : size;

    VkBufferCreateInfo bufferCI{};
    bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCI.size  = allocationSize;
    bufferCI.usage = bufferUsage;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = memoryUsage;
    allocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT; // persistent mapping for CPU-visible buffers

    AllocatedBuffer result;
    result.size = allocationSize;

    checkVk(vmaCreateBuffer(m_allocator, &bufferCI, &allocCI, &result.buffer, &result.allocation, &result.info),
            "Allocator: vmaCreateBuffer");

    return result;
}

void Allocator::destroy(AllocatedBuffer &buffer)
{
    if (buffer.buffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(m_allocator, buffer.buffer, buffer.allocation);
    }
    buffer.buffer     = VK_NULL_HANDLE;
    buffer.allocation = nullptr;
    buffer.info       = {};
    buffer.size       = 0;
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

AllocatedImage Allocator::createImage(const ImageConfig &cfg)
{
    VkExternalMemoryImageCreateInfo externalImageCI{};
    if (cfg.externalMemoryHandleType != 0)
    {
        externalImageCI.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        externalImageCI.handleTypes = cfg.externalMemoryHandleType;
    }

    VkImageCreateInfo imageCI{};
    imageCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.pNext         = cfg.externalMemoryHandleType != 0 ? &externalImageCI : nullptr;
    imageCI.flags         = cfg.flags;
    imageCI.imageType     = cfg.imageType;
    imageCI.format        = cfg.format;
    imageCI.extent        = cfg.extent;
    imageCI.mipLevels     = cfg.mipLevels;
    imageCI.arrayLayers   = cfg.arrayLayers;
    imageCI.samples       = cfg.samples;
    imageCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage         = cfg.usage;
    imageCI.initialLayout = cfg.initialLayout;

    AllocatedImage result;
    result.extent      = cfg.extent;
    result.format      = cfg.format;
    result.mipLevels   = cfg.mipLevels;
    result.arrayLayers = cfg.arrayLayers;
    result.samples     = cfg.samples;
    result.externalMemoryHandleType = cfg.externalMemoryHandleType;

    VmaAllocatorInfo allocInfo;
    vmaGetAllocatorInfo(m_allocator, &allocInfo);
    VkDevice device = allocInfo.device;

    if (cfg.externalMemoryHandleType == 0)
    {
        VmaAllocationCreateInfo allocCI{};
        allocCI.usage         = VMA_MEMORY_USAGE_GPU_ONLY;
        allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        checkVk(vmaCreateImage(m_allocator, &imageCI, &allocCI, &result.image, &result.allocation, nullptr),
                "Allocator: vmaCreateImage");
    } else
    {
        // Validate the format/type/usage/handle-type combination before creating, so an
        // unsupported exportable volume reports what was refused instead of an opaque failure.
        m_ctx.checkExternalImageSupport(imageCI, cfg.externalMemoryHandleType);

        checkVk(vkCreateImage(device, &imageCI, nullptr, &result.image), "Allocator: vkCreateImage(exportable)");
        try
        {
            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(device, result.image, &requirements);

            VkMemoryDedicatedAllocateInfo dedicatedCI{};
            dedicatedCI.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
            dedicatedCI.image = result.image;

            VkExportMemoryAllocateInfo exportCI{};
            exportCI.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
            exportCI.pNext       = &dedicatedCI;
            exportCI.handleTypes = cfg.externalMemoryHandleType;

            VkMemoryAllocateInfo memoryCI{};
            memoryCI.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            memoryCI.pNext           = &exportCI;
            memoryCI.allocationSize  = requirements.size;
            memoryCI.memoryTypeIndex = m_ctx.findMemoryType(requirements.memoryTypeBits,
                                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            checkVk(vkAllocateMemory(device, &memoryCI, nullptr, &result.deviceMemory),
                    "Allocator: vkAllocateMemory(exportable)");
            result.allocationSize = requirements.size;
            checkVk(vkBindImageMemory(device, result.image, result.deviceMemory, 0),
                    "Allocator: vkBindImageMemory(exportable)");
        } catch (...)
        {
            if (result.deviceMemory != VK_NULL_HANDLE)
                vkFreeMemory(device, result.deviceMemory, nullptr);
            vkDestroyImage(device, result.image, nullptr);
            throw;
        }
    }

    // Full-range view covering all mips and all layers
    VkImageViewCreateInfo viewCI{};
    viewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image                           = result.image;
    viewCI.viewType                        = cfg.viewType;
    viewCI.format                          = cfg.format;
    viewCI.subresourceRange.aspectMask     = cfg.aspect;
    viewCI.subresourceRange.baseMipLevel   = 0;
    viewCI.subresourceRange.levelCount     = cfg.mipLevels;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount     = cfg.arrayLayers;

    const VkResult viewResult = vkCreateImageView(device, &viewCI, nullptr, &result.view);
    if (viewResult != VK_SUCCESS)
    {
        if (result.isExternal())
        {
            vkDestroyImage(device, result.image, nullptr);
            if (result.deviceMemory != VK_NULL_HANDLE)
                vkFreeMemory(device, result.deviceMemory, nullptr);
        } else
            vmaDestroyImage(m_allocator, result.image, result.allocation);
        throwVkError(viewResult, "Allocator: vkCreateImageView");
    }

    return result;
}

AllocatedImage Allocator::createImage(VkExtent3D extent, VkFormat format, VkImageUsageFlags usage,
                                      VkImageAspectFlags aspect)
{
    return createImage(ImageConfig{
        .extent = extent,
        .format = format,
        .usage  = usage,
        .aspect = aspect,
    });
}

void Allocator::createMipViews(AllocatedImage &image, VkImageAspectFlags aspect)
{
    VmaAllocatorInfo allocInfo;
    vmaGetAllocatorInfo(m_allocator, &allocInfo);
    VkDevice device = allocInfo.device;

    // Infer the view type for per-mip views (cube stays cube, 2D stays 2D). A 3D image has no
    // valid 2D/cube interpretation here, so reject it rather than create invalid views.
    if (image.extent.depth > 1)
    {
        throw std::logic_error("Allocator: createMipViews does not support 3D images");
    }
    VkImageViewType viewType = (image.arrayLayers == 6) ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;

    if (!image.mipViews.empty())
    {
        throw std::logic_error("Allocator: mip views have already been created");
    }

    std::vector<VkImageView> mipViews(image.mipLevels, VK_NULL_HANDLE);
    for (uint32_t mip = 0; mip < image.mipLevels; ++mip)
    {
        VkImageViewCreateInfo viewCI{};
        viewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCI.image                           = image.image;
        viewCI.viewType                        = viewType;
        viewCI.format                          = image.format;
        viewCI.subresourceRange.aspectMask     = aspect;
        viewCI.subresourceRange.baseMipLevel   = mip;
        viewCI.subresourceRange.levelCount     = 1;
        viewCI.subresourceRange.baseArrayLayer = 0;
        viewCI.subresourceRange.layerCount     = image.arrayLayers;

        const VkResult result = vkCreateImageView(device, &viewCI, nullptr, &mipViews[mip]);
        if (result != VK_SUCCESS)
        {
            for (VkImageView view : mipViews)
            {
                if (view != VK_NULL_HANDLE)
                {
                    vkDestroyImageView(device, view, nullptr);
                }
            }
            throwVkError(result, "Allocator: vkCreateImageView(mip)");
        }
    }
    image.mipViews = std::move(mipViews);
}

void Allocator::createLayerViews(AllocatedImage &image, VkImageAspectFlags aspect)
{
    VmaAllocatorInfo allocInfo;
    vmaGetAllocatorInfo(m_allocator, &allocInfo);
    const VkDevice device = allocInfo.device;
    // Layer views are single-layer 2D views; a 3D image has depth slices, not array layers.
    if (image.extent.depth > 1)
        throw std::logic_error("Allocator: createLayerViews does not support 3D images");
    if (!image.layerViews.empty())
        throw std::runtime_error("Allocator: layer views already exist");
    image.layerViews.resize(image.arrayLayers, VK_NULL_HANDLE);
    for (uint32_t layer = 0; layer < image.arrayLayers; ++layer)
    {
        VkImageViewCreateInfo ci{};
        ci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image                           = image.image;
        ci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        ci.format                          = image.format;
        ci.subresourceRange.aspectMask     = aspect;
        ci.subresourceRange.baseMipLevel   = 0;
        // An attachment view must resolve to a single mip level.
        ci.subresourceRange.levelCount     = 1;
        ci.subresourceRange.baseArrayLayer = layer;
        ci.subresourceRange.layerCount     = 1;
        const VkResult result = vkCreateImageView(device, &ci, nullptr, &image.layerViews[layer]);
        if (result != VK_SUCCESS)
        {
            for (VkImageView view : image.layerViews)
                if (view) vkDestroyImageView(device, view, nullptr);
            image.layerViews.clear();
            throwVkError(result, "Allocator: vkCreateImageView(layer)");
        }
    }
}

void Allocator::destroy(AllocatedImage &image)
{
    VmaAllocatorInfo allocInfo;
    vmaGetAllocatorInfo(m_allocator, &allocInfo);
    VkDevice device = allocInfo.device;

    for (auto view : image.layerViews)
    {
        if (view) vkDestroyImageView(device, view, nullptr);
    }
    image.layerViews.clear();

    for (auto view : image.mipViews)
    {
        if (view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device, view, nullptr);
        }
    }
    image.mipViews.clear();

    if (image.view != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, image.view, nullptr);
    }
    if (image.image != VK_NULL_HANDLE)
    {
        if (image.isExternal())
            vkDestroyImage(device, image.image, nullptr);
        else
            vmaDestroyImage(m_allocator, image.image, image.allocation);
    }
    // Only an external image owns its memory separately; VMA freed it with the image above.
    if (image.deviceMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(device, image.deviceMemory, nullptr);
    }

    image.image          = VK_NULL_HANDLE;
    image.view           = VK_NULL_HANDLE;
    image.allocation     = nullptr;
    image.deviceMemory   = VK_NULL_HANDLE;
    image.allocationSize = 0;
    image.externalMemoryHandleType = static_cast<VkExternalMemoryHandleTypeFlagBits>(0);
    image.extent      = {};
    image.format      = VK_FORMAT_UNDEFINED;
    image.mipLevels   = 1;
    image.arrayLayers = 1;
    image.samples     = VK_SAMPLE_COUNT_1_BIT;
}

} // namespace lr
