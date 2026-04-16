#pragma once

#include "core/vulkan/Allocator.hpp"
#include "core/vulkan/VulkanContext.hpp"

#include <vulkan/vulkan.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace lr
{

class ResourceRegistry
{
public:
    ResourceRegistry(const VulkanContext &ctx, Allocator &allocator, VkExtent2D defaultExtent);
    ~ResourceRegistry();

    ResourceRegistry(const ResourceRegistry &) = delete;
    ResourceRegistry &operator=(const ResourceRegistry &) = delete;

    // -----------------------------------------------------------------------
    // Images
    // -----------------------------------------------------------------------

    // Transient — sized to the swapchain, reallocated on resize.
    // extent {0,0} means "match default extent".
    void registerImage(const std::string &name,
                       VkFormat format,
                       VkImageUsageFlags usage,
                       VkExtent2D extent = {0, 0},
                       VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

    // Persistent — fixed size, never touched on resize.
    // Use for HDRIs, LUTs, shadow maps at fixed resolution.
    void registerPersistentImage(const std::string &name,
                                  VkFormat format,
                                  VkImageUsageFlags usage,
                                  VkExtent2D extent,
                                  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

    // External — wraps an image owned outside the registry (e.g. swapchain).
    // Registers a placeholder so allocateResources skips it and buildBarriers
    // can read the format. The actual VkImage/VkImageView are injected per-frame
    // via FrameGraph::setExternalImage().
    void registerExternalImage(const std::string &name,
                                VkFormat format,
                                VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

    // Persistent cubemap (6 layers, cube-compatible).
    // Optionally creates per-mip VkImageViews (image.mipViews) for compute writes.
    // The full-range view (image.view) is VK_IMAGE_VIEW_TYPE_CUBE for sampling.
    void registerCubemap(const std::string &name,
                          VkFormat format,
                          uint32_t resolution,
                          uint32_t mipLevels,
                          VkImageUsageFlags usage);

    // Register a persistent image AND queue an upload from CPU data.
    // Image is created with TRANSFER_DST_BIT | SAMPLED_BIT automatically.
    // If generateMipmaps is true, also adds TRANSFER_SRC_BIT and runs a blit chain.
    // flushUploads() (called internally by FrameGraph::execute) does the actual transfer.
    void uploadImage(const std::string &name,
                     const void *data,
                     uint32_t width,
                     uint32_t height,
                     VkFormat format,
                     bool generateMipmaps = false);

    // Upload one element of a named image array.
    // Internally this creates a persistent image slot and records that it
    // belongs to arrayName at [index].
    // Slots are currently immutable once uploaded (same as uploadImage).
    void uploadArrayImage(const std::string &arrayName,
                          uint32_t index,
                          const void *data,
                          uint32_t width,
                          uint32_t height,
                          VkFormat format,
                          bool generateMipmaps = false);

    AllocatedImage *getImage(const std::string &name);
    const AllocatedImage *getImage(const std::string &name) const;
    bool hasImage(const std::string &name) const;

    std::vector<const AllocatedImage *> getImageArray(const std::string &arrayName) const;
    bool hasImageArray(const std::string &arrayName) const;

    // Destroys and reallocates all transient images. Call on swapchain resize.
    void rebuild(VkExtent2D newExtent);
    VkExtent2D getExtent() const { return m_defaultExtent; }

    // -----------------------------------------------------------------------
    // Buffers
    // -----------------------------------------------------------------------

    // Dynamic — CPU_TO_GPU, persistently mapped. Written by CPU each frame.
    // Use for uniforms, light data, per-frame scene parameters.
    void registerDynamicBuffer(const std::string &name,
                                VkDeviceSize size,
                                VkBufferUsageFlags usage);

    // Static — GPU_ONLY, no initial data. Use for compute scratch buffers.
    void registerStaticBuffer(const std::string &name,
                               VkDeviceSize size,
                               VkBufferUsageFlags usage);

    // Register a GPU_ONLY buffer AND queue an upload from CPU data.
    // Buffer is created with TRANSFER_DST_BIT automatically.
    // flushUploads() (called internally by FrameGraph::execute) does the actual transfer.
    void uploadBuffer(const std::string &name,
                      const void *data,
                      VkDeviceSize size,
                      VkBufferUsageFlags usage);

    // Overwrite the contents of a dynamic buffer registered via registerDynamicBuffer().
    // data must point to at least size bytes. Call once per frame before fg.execute().
    void updateBuffer(const std::string &name, const void *data, VkDeviceSize size);

    AllocatedBuffer *getBuffer(const std::string &name);
    const AllocatedBuffer *getBuffer(const std::string &name) const;
    bool hasBuffer(const std::string &name) const;

    // -----------------------------------------------------------------------
    // Upload queue — called automatically by FrameGraph::execute()
    // -----------------------------------------------------------------------

    bool hasPendingUploads() const { return !m_pendingUploads.empty(); }
    void flushUploads();

private:
    struct ImageEntry
    {
        AllocatedImage image;
        VkFormat format;
        VkImageUsageFlags usage;
        VkImageAspectFlags aspect;
        VkExtent2D extent;
        bool persistent = false;
        // Extended image properties — used by allocateImageEntry and rebuild
        uint32_t mipLevels = 1;
        uint32_t arrayLayers = 1;
        VkImageCreateFlags imageFlags = 0;
        VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
        bool hasMipViews = false;
        VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    struct BufferEntry
    {
        AllocatedBuffer buffer;
        VkDeviceSize size;
        VkBufferUsageFlags usage;
        VmaMemoryUsage memoryUsage;
    };

    struct PendingUpload
    {
        AllocatedBuffer staging;

        enum class Type { Buffer, Image } type;

        // Buffer upload
        AllocatedBuffer *destBuffer = nullptr;

        // Image upload
        AllocatedImage *destImage = nullptr;
        VkExtent3D imageExtent{};
        bool generateMipmaps = false;
    };

    void allocateImageEntry(const std::string &name, ImageEntry &entry);
    void initCommandPool();

    void setDebugName(VkObjectType objectType, uint64_t objectHandle, const std::string &name);

    const VulkanContext &m_ctx;
    Allocator &m_allocator;
    VkExtent2D m_defaultExtent;

    std::unordered_map<std::string, ImageEntry> m_images;
    std::unordered_map<std::string, std::vector<std::string>> m_imageArrays;
    std::unordered_map<std::string, BufferEntry> m_buffers;

    std::vector<PendingUpload> m_pendingUploads;

    VkCommandPool m_uploadPool = VK_NULL_HANDLE;
};

}  // namespace lr
