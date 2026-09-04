#include "ResourceRegistry.hpp"
#include "core/vulkan/VkResultUtils.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace
{
uint32_t computeMipLevels(uint32_t width, uint32_t height)
{
    return static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
}
} // namespace

namespace lr
{

ResourceRegistry::ResourceRegistry(const VulkanContext &ctx, Allocator &allocator, VkExtent2D defaultExtent)
    : m_ctx(ctx), m_allocator(allocator), m_defaultExtent(defaultExtent)
{
    initCommandPool();
}

ResourceRegistry::~ResourceRegistry()
{
    // Staging buffers should have been freed by flushUploads(), but clean up
    // anything left over if shutdown happens before the first execute()
    for (auto &upload : m_pendingUploads)
    {
        m_allocator.destroy(upload.staging);
    }

    for (auto &[name, entry] : m_images)
    {
        m_allocator.destroy(entry.image);
    }

    for (auto &[name, entry] : m_buffers)
    {
        m_allocator.destroy(entry.buffer);
    }

    vkDestroyCommandPool(m_ctx.getDevice(), m_uploadPool, nullptr);
}

void ResourceRegistry::initCommandPool()
{
    VkCommandPoolCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = static_cast<uint32_t>(m_ctx.getGraphicsQueueFamily());
    ci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

    checkVk(vkCreateCommandPool(m_ctx.getDevice(), &ci, nullptr, &m_uploadPool),
            "ResourceRegistry: vkCreateCommandPool");
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

void ResourceRegistry::registerImage(const std::string &name, VkFormat format, VkImageUsageFlags usage,
                                     VkExtent2D extent, VkImageAspectFlags aspect)
{
    if (m_images.count(name))
    {
        throw std::runtime_error("ResourceRegistry: duplicate image '" + name + "'");
    }

    ImageEntry entry{};
    entry.format     = format;
    entry.usage      = usage;
    entry.aspect     = aspect;
    entry.extent     = (extent.width == 0 || extent.height == 0) ? m_defaultExtent : extent;
    entry.persistent = false;

    allocateImageEntry(name, entry);
    VkExtent2D ext = entry.extent;
    m_images.emplace(name, std::move(entry));
    spdlog::debug("ResourceRegistry: image '{}' ({}x{})", name, ext.width, ext.height);
}

void ResourceRegistry::registerExternalImage(const std::string &name, VkFormat format, VkImageAspectFlags aspect)
{
    if (m_images.count(name))
    {
        throw std::runtime_error("ResourceRegistry: duplicate image '" + name + "'");
    }

    // Placeholder — no VMA allocation. VkImage/VkImageView are null until
    // patched each frame by FrameGraph::setExternalImage().
    ImageEntry entry{};
    entry.format       = format;
    entry.image.format = format; // mirror so getImage()->format is valid
    entry.usage        = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    entry.aspect       = aspect;
    entry.extent       = m_defaultExtent;
    entry.persistent   = true;
    entry.external     = true;
    entry.mipLevels    = 1;
    entry.arrayLayers  = 1;
    // entry.image left zero-initialised (VK_NULL_HANDLE handles)

    m_images.emplace(name, std::move(entry));
    spdlog::debug("ResourceRegistry: external image '{}' registered", name);
    // No debug name — external images have no VkImage allocation here;
    // the swapchain image is owned and named by the presentation engine.
}

void ResourceRegistry::registerPersistentImage(const std::string &name, VkFormat format, VkImageUsageFlags usage,
                                               VkExtent2D extent, VkImageAspectFlags aspect)
{
    if (m_images.count(name))
    {
        throw std::runtime_error("ResourceRegistry: duplicate image '" + name + "'");
    }

    ImageEntry entry{};
    entry.format     = format;
    entry.usage      = usage;
    entry.aspect     = aspect;
    entry.extent     = extent;
    entry.persistent = true;

    allocateImageEntry(name, entry);
    m_images.emplace(name, std::move(entry));
    spdlog::debug("ResourceRegistry: persistent image '{}' ({}x{})", name, extent.width, extent.height);
}

void ResourceRegistry::registerCubemap(const std::string &name, VkFormat format, uint32_t resolution,
                                       uint32_t mipLevels, VkImageUsageFlags usage)
{
    if (m_images.count(name))
    {
        throw std::runtime_error("ResourceRegistry: duplicate image '" + name + "'");
    }

    ImageEntry entry{};
    entry.format      = format;
    entry.usage       = usage;
    entry.aspect      = VK_IMAGE_ASPECT_COLOR_BIT;
    entry.extent      = {resolution, resolution};
    entry.persistent  = true;
    entry.mipLevels   = mipLevels;
    entry.arrayLayers = 6;
    entry.imageFlags  = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    entry.viewType    = VK_IMAGE_VIEW_TYPE_CUBE;
    entry.hasMipViews = (mipLevels > 1);

    allocateImageEntry(name, entry);
    m_images.emplace(name, std::move(entry));
    spdlog::debug("ResourceRegistry: cubemap '{}' ({}x{}, {} mips)", name, resolution, resolution, mipLevels);
}

void ResourceRegistry::uploadImage(const std::string &name, const void *data, uint32_t width, uint32_t height,
                                   VkFormat format, bool generateMipmaps)
{
    queueImageUpload(name, data, width, height, format, generateMipmaps, ImageUploadMode::Create);
}

void ResourceRegistry::replaceUploadedImage(const std::string &name, const void *data, uint32_t width, uint32_t height,
                                            VkFormat format, bool generateMipmaps)
{
    queueImageUpload(name, data, width, height, format, generateMipmaps, ImageUploadMode::Replace);
}

void ResourceRegistry::queueImageUpload(const std::string &name, const void *data, uint32_t width, uint32_t height,
                                        VkFormat format, bool generateMipmaps, ImageUploadMode mode)
{
    auto existing = m_images.find(name);
    if (mode == ImageUploadMode::Create && existing != m_images.end())
    {
        throw std::runtime_error("ResourceRegistry: duplicate image '" + name + "'");
    }
    if (mode == ImageUploadMode::Replace && existing == m_images.end())
    {
        throw std::runtime_error("ResourceRegistry: image '" + name + "' not found");
    }
    if (mode == ImageUploadMode::Replace && (!existing->second.persistent || existing->second.external))
    {
        throw std::runtime_error("ResourceRegistry: image '" + name + "' is not a replaceable uploaded image");
    }

    for (const auto &upload : m_pendingUploads)
    {
        if (upload.type == PendingUpload::Type::Image && upload.resourceName == name)
        {
            throw std::runtime_error("ResourceRegistry: image '" + name + "' already has a pending upload");
        }
    }

    VkDeviceSize bpp = 4; // default RGBA8
    if (format == VK_FORMAT_R16G16B16A16_SFLOAT)
    {
        bpp = 8;
    }
    if (format == VK_FORMAT_R32G32B32A32_SFLOAT)
    {
        bpp = 16;
    }
    if (format == VK_FORMAT_R8_UNORM)
    {
        bpp = 1;
    }
    if (format == VK_FORMAT_R8G8_UNORM)
    {
        bpp = 2;
    }

    const uint32_t    mipLevels = generateMipmaps ? computeMipLevels(width, height) : 1;
    VkImageUsageFlags usage     = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (generateMipmaps)
    {
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }

    ImageEntry prepared{};
    prepared.format        = format;
    prepared.usage         = usage;
    prepared.aspect        = VK_IMAGE_ASPECT_COLOR_BIT;
    prepared.extent        = {width, height};
    prepared.persistent    = true;
    prepared.mipLevels     = mipLevels;
    prepared.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    prepared.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    AllocatedBuffer staging{};
    try
    {
        // Prepare every fallible allocation before changing the registry.
        allocateImageEntry(name, prepared);
        const VkDeviceSize size = static_cast<VkDeviceSize>(width) * height * bpp;
        staging = m_allocator.createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
        std::memcpy(staging.info.pMappedData, data, size);

        PendingUpload pending{
            .staging      = staging,
            .resourceName = name,
            .type         = PendingUpload::Type::Image,
        };
        m_pendingUploads.reserve(m_pendingUploads.size() + 1);

        if (mode == ImageUploadMode::Create)
        {
            m_images.emplace(name, std::move(prepared));
        } else
        {
            AllocatedImage oldImage = std::move(existing->second.image);
            existing->second        = std::move(prepared);
            m_allocator.destroy(oldImage);
        }

        // Capacity was reserved before committing the image, so this move cannot reallocate.
        m_pendingUploads.push_back(std::move(pending));
    } catch (...)
    {
        if (staging.buffer != VK_NULL_HANDLE)
        {
            m_allocator.destroy(staging);
        }
        if (prepared.image.image != VK_NULL_HANDLE)
        {
            m_allocator.destroy(prepared.image);
        }
        throw;
    }

    spdlog::debug("ResourceRegistry: {} image '{}' ({}x{}, {} mips)",
                  mode == ImageUploadMode::Create ? "queued" : "replaced", name, width, height, mipLevels);
}

void ResourceRegistry::uploadArrayImage(const std::string &arrayName, uint32_t index, const void *data, uint32_t width,
                                        uint32_t height, VkFormat format, bool generateMipmaps)
{
    if (arrayName.empty())
    {
        throw std::invalid_argument("ResourceRegistry: image array name cannot be empty");
    }

    const std::string slotName = arrayName + "[" + std::to_string(index) + "]";

    // Reuse the existing single-image upload path per slot.
    uploadImage(slotName, data, width, height, format, generateMipmaps);

    auto &slots = m_imageArrays[arrayName];
    if (slots.size() <= index)
    {
        slots.resize(static_cast<size_t>(index) + 1);
    }

    slots[index] = slotName;
    spdlog::debug("ResourceRegistry: image array '{}' slot {} -> '{}'", arrayName, index, slotName);
}

AllocatedImage *ResourceRegistry::getImage(const std::string &name)
{
    auto it = m_images.find(name);
    return (it != m_images.end()) ? &it->second.image : nullptr;
}

const AllocatedImage *ResourceRegistry::getImage(const std::string &name) const
{
    auto it = m_images.find(name);
    return (it != m_images.end()) ? &it->second.image : nullptr;
}

bool ResourceRegistry::hasImage(const std::string &name) const { return m_images.count(name) > 0; }

void ResourceRegistry::validateImage(const std::string &name, VkFormat format, VkImageUsageFlags requiredUsage,
                                     VkExtent2D extent, VkImageAspectFlags aspect) const
{
    const auto it = m_images.find(name);
    if (it == m_images.end())
    {
        throw std::runtime_error("ResourceRegistry: image '" + name + "' not found");
    }

    const ImageEntry &entry = it->second;
    if (entry.format != format)
    {
        throw std::runtime_error("FrameGraph: registered image '" + name + "' has an incompatible format");
    }
    if (entry.extent.width != extent.width || entry.extent.height != extent.height)
    {
        throw std::runtime_error("FrameGraph: registered image '" + name + "' has an incompatible extent");
    }
    if (entry.aspect != aspect)
    {
        throw std::runtime_error("FrameGraph: registered image '" + name + "' has an incompatible aspect");
    }
    if ((entry.usage & requiredUsage) != requiredUsage)
    {
        throw std::runtime_error("FrameGraph: registered image '" + name + "' is missing required usage flags");
    }
}

void ResourceRegistry::validateImageUsage(const std::string &name, VkImageUsageFlags requiredUsage) const
{
    if (const auto image = m_images.find(name); image != m_images.end())
    {
        if ((image->second.usage & requiredUsage) != requiredUsage)
        {
            throw std::runtime_error("FrameGraph: registered image '" + name + "' is missing required usage flags");
        }
        return;
    }

    const auto array = m_imageArrays.find(name);
    if (array == m_imageArrays.end())
    {
        throw std::runtime_error("FrameGraph: image or image array '" + name + "' not found");
    }
    for (const std::string &slotName : array->second)
    {
        if (slotName.empty())
        {
            continue;
        }
        const ImageEntry &slot = m_images.at(slotName);
        if ((slot.usage & requiredUsage) != requiredUsage)
        {
            throw std::runtime_error("FrameGraph: image array '" + name + "' is missing required usage flags");
        }
    }
}

std::vector<const AllocatedImage *> ResourceRegistry::getImageArray(const std::string &arrayName) const
{
    std::vector<const AllocatedImage *> result;

    auto it = m_imageArrays.find(arrayName);
    if (it == m_imageArrays.end())
    {
        return result;
    }

    result.reserve(it->second.size());
    for (const auto &slotName : it->second)
    {
        if (slotName.empty())
        {
            result.push_back(nullptr);
        } else
        {
            result.push_back(getImage(slotName));
        }
    }

    return result;
}

std::vector<VkImageLayout> ResourceRegistry::getImageArrayLayouts(const std::string &arrayName) const
{
    std::vector<VkImageLayout> result;
    const auto                 it = m_imageArrays.find(arrayName);
    if (it == m_imageArrays.end())
    {
        return result;
    }

    result.reserve(it->second.size());
    for (const std::string &slotName : it->second)
    {
        result.push_back(slotName.empty() ? VK_IMAGE_LAYOUT_UNDEFINED : getImageLayout(slotName));
    }
    return result;
}

void ResourceRegistry::setImageArrayLayout(const std::string &arrayName, VkImageLayout layout)
{
    const auto it = m_imageArrays.find(arrayName);
    if (it == m_imageArrays.end())
    {
        return;
    }
    for (const std::string &slotName : it->second)
    {
        if (!slotName.empty())
        {
            setImageLayout(slotName, layout);
        }
    }
}

bool ResourceRegistry::hasImageArray(const std::string &arrayName) const { return m_imageArrays.count(arrayName) > 0; }

void ResourceRegistry::rebuild(VkExtent2D newExtent)
{
    m_defaultExtent = newExtent;

    for (auto &[name, entry] : m_images)
    {
        if (entry.external)
        {
            entry.extent        = newExtent;
            entry.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            continue;
        }

        if (entry.persistent)
        {
            continue;
        }

        m_allocator.destroy(entry.image);
        entry.extent        = newExtent;
        entry.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        allocateImageEntry(name, entry);
        spdlog::debug("ResourceRegistry: rebuilt image '{}'", name);
    }
}

void ResourceRegistry::allocateImageEntry(const std::string &name, ImageEntry &entry)
{
    VkExtent3D extent3d{entry.extent.width, entry.extent.height, 1};
    entry.image = m_allocator.createImage(ImageConfig{
        .extent        = extent3d,
        .format        = entry.format,
        .usage         = entry.usage,
        .aspect        = entry.aspect,
        .mipLevels     = entry.mipLevels,
        .arrayLayers   = entry.arrayLayers,
        .flags         = entry.imageFlags,
        .viewType      = entry.viewType,
        .initialLayout = entry.initialLayout,
    });
    if (entry.hasMipViews)
    {
        m_allocator.createMipViews(entry.image, entry.aspect);
    }
    setDebugName(VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(entry.image.image), name);
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

void ResourceRegistry::registerDynamicBuffer(const std::string &name, VkDeviceSize size, VkBufferUsageFlags usage)
{
    if (m_buffers.count(name))
    {
        throw std::runtime_error("ResourceRegistry: duplicate buffer '" + name + "'");
    }

    BufferEntry entry{};
    entry.size        = size;
    entry.usage       = usage;
    entry.memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    entry.buffer      = m_allocator.createBuffer(size, usage, VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_buffers.emplace(name, std::move(entry));
    setDebugName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(m_buffers.at(name).buffer.buffer), name);
    spdlog::debug("ResourceRegistry: dynamic buffer '{}' ({} bytes)", name, size);
}

void ResourceRegistry::registerStaticBuffer(const std::string &name, VkDeviceSize size, VkBufferUsageFlags usage)
{
    if (m_buffers.count(name))
    {
        throw std::runtime_error("ResourceRegistry: duplicate buffer '" + name + "'");
    }

    BufferEntry entry{};
    entry.size        = size;
    entry.usage       = usage;
    entry.memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY;
    entry.buffer      = m_allocator.createBuffer(size, usage, VMA_MEMORY_USAGE_GPU_ONLY);
    m_buffers.emplace(name, std::move(entry));
    setDebugName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(m_buffers.at(name).buffer.buffer), name);
    spdlog::debug("ResourceRegistry: static buffer '{}' ({} bytes)", name, size);
}

void ResourceRegistry::uploadBuffer(const std::string &name, const void *data, VkDeviceSize size,
                                    VkBufferUsageFlags usage)
{
    // Register as GPU_ONLY with transfer dst
    registerStaticBuffer(name, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Create staging buffer and copy data in
    AllocatedBuffer staging =
        m_allocator.createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    std::memcpy(staging.info.pMappedData, data, size);

    PendingUpload upload{};
    upload.staging      = staging;
    upload.type         = PendingUpload::Type::Buffer;
    upload.resourceName = name;
    m_pendingUploads.push_back(upload);

    spdlog::debug("ResourceRegistry: queued buffer upload '{}'", name);
}

AllocatedBuffer *ResourceRegistry::getBuffer(const std::string &name)
{
    auto it = m_buffers.find(name);
    return (it != m_buffers.end()) ? &it->second.buffer : nullptr;
}

const AllocatedBuffer *ResourceRegistry::getBuffer(const std::string &name) const
{
    auto it = m_buffers.find(name);
    return (it != m_buffers.end()) ? &it->second.buffer : nullptr;
}

bool ResourceRegistry::hasBuffer(const std::string &name) const { return m_buffers.count(name) > 0; }

void ResourceRegistry::updateBuffer(const std::string &name, const void *data, VkDeviceSize size)
{
    auto it = m_buffers.find(name);
    if (it == m_buffers.end())
    {
        throw std::runtime_error("ResourceRegistry: updateBuffer '" + name + "' not found");
    }
    if (it->second.memoryUsage != VMA_MEMORY_USAGE_CPU_TO_GPU)
    {
        throw std::runtime_error("ResourceRegistry: updateBuffer '" + name +
                                 "' is not a dynamic buffer — use registerDynamicBuffer() to create updatable buffers");
    }
    if (size > it->second.size)
    {
        throw std::runtime_error("ResourceRegistry: updateBuffer '" + name + "' size overflow");
    }
    std::memcpy(it->second.buffer.info.pMappedData, data, size);
}

void ResourceRegistry::reuploadBuffer(const std::string &name, const void *data, VkDeviceSize size)
{
    auto it = m_buffers.find(name);
    if (it == m_buffers.end())
    {
        throw std::runtime_error("ResourceRegistry: reuploadBuffer '" + name + "' not found");
    }
    if (it->second.memoryUsage != VMA_MEMORY_USAGE_GPU_ONLY)
    {
        throw std::runtime_error("ResourceRegistry: reuploadBuffer '" + name +
                                 "' is not a static buffer — use updateBuffer() for dynamic buffers");
    }
    if (size > it->second.size)
    {
        throw std::runtime_error("ResourceRegistry: reuploadBuffer '" + name + "' size overflow");
    }

    AllocatedBuffer staging =
        m_allocator.createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    std::memcpy(staging.info.pMappedData, data, size);

    PendingUpload upload{};
    upload.staging      = staging;
    upload.type         = PendingUpload::Type::Buffer;
    upload.resourceName = name;
    m_pendingUploads.push_back(upload);
}

// ---------------------------------------------------------------------------
// Upload flush
// ---------------------------------------------------------------------------

void ResourceRegistry::flushUploads()
{
    if (m_pendingUploads.empty())
    {
        return;
    }

    spdlog::debug("ResourceRegistry: flushing {} upload(s)...", m_pendingUploads.size());

    VkDevice device = m_ctx.getDevice();

    // Allocate a one-time command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = m_uploadPool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    checkVk(vkAllocateCommandBuffers(device, &allocInfo, &cmd), "ResourceRegistry: vkAllocateCommandBuffers");

    VkFence fence     = VK_NULL_HANDLE;
    bool    submitted = false;
    try
    {

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        checkVk(vkBeginCommandBuffer(cmd, &beginInfo), "ResourceRegistry: vkBeginCommandBuffer");

        for (auto &upload : m_pendingUploads)
        {
            if (upload.type == PendingUpload::Type::Buffer)
            {
                AllocatedBuffer *dest = getBuffer(upload.resourceName);
                VkBufferCopy     region{};
                region.size = dest->size;
                vkCmdCopyBuffer(cmd, upload.staging.buffer, dest->buffer, 1, &region);
            } else
            {
                AllocatedImage *img    = getImage(upload.resourceName);
                const uint32_t  layers = img->arrayLayers;
                const uint32_t  mips   = img->mipLevels;

                // Transition mip 0: UNDEFINED → TRANSFER_DST_OPTIMAL
                VkImageMemoryBarrier2 barrier{};
                barrier.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                barrier.srcStageMask     = VK_PIPELINE_STAGE_2_NONE;
                barrier.srcAccessMask    = VK_ACCESS_2_NONE;
                barrier.dstStageMask     = VK_PIPELINE_STAGE_2_COPY_BIT;
                barrier.dstAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                barrier.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.image            = img->image;
                barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};

                VkDependencyInfo dep{};
                dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers    = &barrier;
                vkCmdPipelineBarrier2(cmd, &dep);

                // Copy CPU data into mip 0
                VkBufferImageCopy region{};
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers};
                region.imageExtent      = img->extent;
                vkCmdCopyBufferToImage(cmd, upload.staging.buffer, img->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                       &region);

                if (mips > 1)
                {
                    // Blit chain: mip 0 → 1 → 2 → ... → N-1
                    int32_t mipW = static_cast<int32_t>(img->extent.width);
                    int32_t mipH = static_cast<int32_t>(img->extent.height);

                    for (uint32_t mip = 1; mip < mips; ++mip)
                    {
                        VkImageMemoryBarrier2 barriers[2]{};

                        barriers[0].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                        barriers[0].srcStageMask     = VK_PIPELINE_STAGE_2_COPY_BIT;
                        barriers[0].srcAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                        barriers[0].dstStageMask     = VK_PIPELINE_STAGE_2_BLIT_BIT;
                        barriers[0].dstAccessMask    = VK_ACCESS_2_TRANSFER_READ_BIT;
                        barriers[0].oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        barriers[0].newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        barriers[0].image            = img->image;
                        barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, 1, 0, layers};

                        barriers[1].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                        barriers[1].srcStageMask     = VK_PIPELINE_STAGE_2_NONE;
                        barriers[1].srcAccessMask    = VK_ACCESS_2_NONE;
                        barriers[1].dstStageMask     = VK_PIPELINE_STAGE_2_BLIT_BIT;
                        barriers[1].dstAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                        barriers[1].oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
                        barriers[1].newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        barriers[1].image            = img->image;
                        barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, layers};

                        VkDependencyInfo blitDep{};
                        blitDep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                        blitDep.imageMemoryBarrierCount = 2;
                        blitDep.pImageMemoryBarriers    = barriers;
                        vkCmdPipelineBarrier2(cmd, &blitDep);

                        int32_t nextW = std::max(mipW >> 1, 1);
                        int32_t nextH = std::max(mipH >> 1, 1);

                        VkImageBlit2 blit{};
                        blit.sType          = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
                        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, 0, layers};
                        blit.srcOffsets[0]  = {0, 0, 0};
                        blit.srcOffsets[1]  = {mipW, mipH, 1};
                        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, layers};
                        blit.dstOffsets[0]  = {0, 0, 0};
                        blit.dstOffsets[1]  = {nextW, nextH, 1};

                        VkBlitImageInfo2 blitInfo{};
                        blitInfo.sType          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
                        blitInfo.srcImage       = img->image;
                        blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        blitInfo.dstImage       = img->image;
                        blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        blitInfo.regionCount    = 1;
                        blitInfo.pRegions       = &blit;
                        blitInfo.filter         = VK_FILTER_LINEAR;
                        vkCmdBlitImage2(cmd, &blitInfo);

                        mipW = nextW;
                        mipH = nextH;
                    }

                    // Final transitions:
                    // mips 0..N-2 are in TRANSFER_SRC_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
                    // mip  N-1    is in TRANSFER_DST_OPTIMAL  → SHADER_READ_ONLY_OPTIMAL
                    VkImageMemoryBarrier2 finalBarriers[2]{};

                    finalBarriers[0].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    finalBarriers[0].srcStageMask     = VK_PIPELINE_STAGE_2_BLIT_BIT;
                    finalBarriers[0].srcAccessMask    = VK_ACCESS_2_TRANSFER_READ_BIT;
                    finalBarriers[0].dstStageMask     = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                    finalBarriers[0].dstAccessMask    = VK_ACCESS_2_SHADER_READ_BIT;
                    finalBarriers[0].oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    finalBarriers[0].newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    finalBarriers[0].image            = img->image;
                    finalBarriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips - 1, 0, layers};

                    finalBarriers[1].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    finalBarriers[1].srcStageMask     = VK_PIPELINE_STAGE_2_BLIT_BIT;
                    finalBarriers[1].srcAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    finalBarriers[1].dstStageMask     = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                    finalBarriers[1].dstAccessMask    = VK_ACCESS_2_SHADER_READ_BIT;
                    finalBarriers[1].oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    finalBarriers[1].newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    finalBarriers[1].image            = img->image;
                    finalBarriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mips - 1, 1, 0, layers};

                    VkDependencyInfo finalDep{};
                    finalDep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    finalDep.imageMemoryBarrierCount = 2;
                    finalDep.pImageMemoryBarriers    = finalBarriers;
                    vkCmdPipelineBarrier2(cmd, &finalDep);
                } else
                {
                    // No mipmaps — transition mip 0 directly to SHADER_READ_ONLY_OPTIMAL
                    barrier.srcStageMask     = VK_PIPELINE_STAGE_2_COPY_BIT;
                    barrier.srcAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    barrier.dstStageMask     = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                    barrier.dstAccessMask    = VK_ACCESS_2_SHADER_READ_BIT;
                    barrier.oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    barrier.newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
                    vkCmdPipelineBarrier2(cmd, &dep);
                }
            }
        }

        checkVk(vkEndCommandBuffer(cmd), "ResourceRegistry: vkEndCommandBuffer");

        // Submit and wait
        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        checkVk(vkCreateFence(device, &fenceCI, nullptr, &fence), "ResourceRegistry: vkCreateFence");

        VkCommandBufferSubmitInfo cmdSubmit{};
        cmdSubmit.sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cmdSubmit.commandBuffer = cmd;

        VkSubmitInfo2 submitInfo{};
        submitInfo.sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos    = &cmdSubmit;

        checkVk(vkQueueSubmit2(m_ctx.getGraphicsQueue(), 1, &submitInfo, fence), "ResourceRegistry: vkQueueSubmit2");
        submitted = true;
        checkVk(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "ResourceRegistry: vkWaitForFences");
    } catch (...)
    {
        // A failed wait does not prove that submitted GPU work has stopped. Quiesce
        // the device before releasing its command buffer and fence.
        if (submitted)
        {
            (void)vkDeviceWaitIdle(device);
        }
        if (fence != VK_NULL_HANDLE)
        {
            vkDestroyFence(device, fence, nullptr);
        }
        vkFreeCommandBuffers(device, m_uploadPool, 1, &cmd);
        throw;
    }

    // Cleanup
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, m_uploadPool, 1, &cmd);

    for (auto &upload : m_pendingUploads)
    {
        m_allocator.destroy(upload.staging);
        if (upload.type == PendingUpload::Type::Image)
        {
            setImageLayout(upload.resourceName, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }

    m_pendingUploads.clear();

    spdlog::debug("ResourceRegistry: uploads complete");
}

// ---------------------------------------------------------------------------
// Debug utils
// ---------------------------------------------------------------------------

VkImageLayout ResourceRegistry::getImageLayout(const std::string &name) const
{
    auto it = m_images.find(name);
    return (it != m_images.end()) ? it->second.currentLayout : VK_IMAGE_LAYOUT_UNDEFINED;
}

void ResourceRegistry::setImageLayout(const std::string &name, VkImageLayout layout)
{
    auto it = m_images.find(name);
    if (it != m_images.end())
    {
        it->second.currentLayout = layout;
    }
}

void ResourceRegistry::setDebugName(VkObjectType objectType, uint64_t objectHandle, const std::string &name)
{
    m_ctx.setDebugName(objectType, objectHandle, name);
}
} // namespace lr
