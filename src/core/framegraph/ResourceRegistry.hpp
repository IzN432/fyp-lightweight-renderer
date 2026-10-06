#pragma once

#include "ExtentSpec.hpp"
#include "core/vulkan/Allocator.hpp"
#include "core/vulkan/VulkanContext.hpp"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace lr
{

class ResourceRegistry
{
public:
    // Compiled graphs use the same allocator for private implementation images such as MSAA targets.
    // These images are deliberately not registered by name: user-facing resources stay single-sampled.
    Allocator &allocator() { return m_allocator; }
    ResourceRegistry(const VulkanContext &ctx, Allocator &allocator, VkExtent2D defaultExtent);
    ~ResourceRegistry();

    ResourceRegistry(const ResourceRegistry &)            = delete;
    ResourceRegistry &operator=(const ResourceRegistry &) = delete;

    // -----------------------------------------------------------------------
    // Images
    // -----------------------------------------------------------------------

    // Transient — its symbolic extent is resolved against the swapchain on allocation and resize.
    void registerImage(const std::string &name, VkFormat format, VkImageUsageFlags usage,
                       ExtentSpec         extent = ExtentSpec::swapchain(),
                       VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

    // Persistent — fixed size, never touched on resize.
    // Use for HDRIs, LUTs, shadow maps at fixed resolution.
    void registerPersistentImage(const std::string &name, VkFormat format, VkImageUsageFlags usage, VkExtent2D extent,
                                 VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

    // External — wraps an image owned outside the registry (e.g. swapchain).
    // Registers a placeholder so allocateResources skips it and buildBarriers
    // can read the format. The actual VkImage/VkImageView are injected per-frame
    // via FrameGraph::setExternalImage().
    void registerExternalImage(const std::string &name, VkFormat format,
                               VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

    // Persistent cubemap (6 layers, cube-compatible).
    // Optionally creates per-mip VkImageViews (image.mipViews) for compute writes.
    // The full-range view (image.view) is VK_IMAGE_VIEW_TYPE_CUBE for sampling.
    void registerCubemap(const std::string &name, VkFormat format, uint32_t resolution, uint32_t mipLevels,
                         VkImageUsageFlags usage);

    // Register a persistent image AND queue an upload from CPU data.
    // Image is created with TRANSFER_DST_BIT | SAMPLED_BIT automatically.
    // If generateMipmaps is true, also adds TRANSFER_SRC_BIT and runs a blit chain.
    // flushUploads() (called internally by FrameGraph::execute) does the actual transfer.
    void uploadImage(const std::string &name, const void *data, uint32_t width, uint32_t height, VkFormat format,
                     bool generateMipmaps = false);

    // Replace an uploaded persistent image while preserving its registry name. The old image is
    // retired, not destroyed: it lives until every frame that may still use it has completed.
    void replaceUploadedImage(const std::string &name, const void *data, uint32_t width, uint32_t height,
                              VkFormat format, bool generateMipmaps = false);

    // Upload one element of a named image array.
    // Internally this creates a persistent image slot and records that it
    // belongs to arrayName at [index].
    // Slots are currently immutable once uploaded (same as uploadImage).
    void uploadArrayImage(const std::string &arrayName, uint32_t index, const void *data, uint32_t width,
                          uint32_t height, VkFormat format, bool generateMipmaps = false);

    AllocatedImage       *getImage(const std::string &name);
    const AllocatedImage *getImage(const std::string &name) const;
    VkExtent2D            getImageExtent(const std::string &name) const;
    bool                  hasImage(const std::string &name) const;
    void validateImage(const std::string &name, VkFormat format, VkImageUsageFlags requiredUsage, ExtentSpec extent,
                       VkImageAspectFlags aspect) const;
    void validateImageUsage(const std::string &name, VkImageUsageFlags requiredUsage) const;

    VkImageLayout getImageLayout(const std::string &name) const;
    void          setImageLayout(const std::string &name, VkImageLayout layout);

    std::vector<const AllocatedImage *> getImageArray(const std::string &arrayName) const;
    std::vector<VkImageLayout>          getImageArrayLayouts(const std::string &arrayName) const;
    void                                setImageArrayLayout(const std::string &arrayName, VkImageLayout layout);
    bool                                hasImageArray(const std::string &arrayName) const;

    // Every registered image, image array and buffer name (unordered).
    std::vector<std::string> names() const;

    // Destroys and reallocates all transient images. Call on swapchain resize.
    void       rebuild(VkExtent2D newExtent);
    VkExtent2D getExtent() const { return m_defaultExtent; }

    // -----------------------------------------------------------------------
    // Buffers
    // -----------------------------------------------------------------------

    // Dynamic — CPU_TO_GPU, persistently mapped. Written by CPU each frame.
    // Use for uniforms, light data, per-frame scene parameters. Holds one copy per frame in flight
    // (see setFramesInFlight) so the CPU never overwrites data a still-executing frame reads; GPU
    // writes to dynamic buffers aren't supported (use a static buffer).
    void registerDynamicBuffer(const std::string &name, VkDeviceSize size, VkBufferUsageFlags usage);

    // Static — GPU_ONLY, no initial data. Use for compute scratch buffers.
    void registerStaticBuffer(const std::string &name, VkDeviceSize size, VkBufferUsageFlags usage);

    // Register a GPU_ONLY buffer AND queue an upload from CPU data.
    // Buffer is created with TRANSFER_DST_BIT automatically.
    // flushUploads() (called internally by FrameGraph::execute) does the actual transfer.
    void uploadBuffer(const std::string &name, const void *data, VkDeviceSize size, VkBufferUsageFlags usage);

    // Overwrite the first `size` bytes of a dynamic buffer registered via registerDynamicBuffer().
    // Writes this frame's copy; the contents persist (later frames' copies are brought up to date
    // as they come round), so data written once before the first frame is seen by every frame.
    void updateBuffer(const std::string &name, const void *data, VkDeviceSize size);

    // Queue a staging upload to an existing static buffer (created via uploadBuffer()).
    // The copy is folded into the next flushUploads() call (i.e. the next frame's execute).
    void reuploadBuffer(const std::string &name, const void *data, VkDeviceSize size);

    // Reallocates a static buffer previously created via uploadBuffer(), sized for `size` (unlike
    // reuploadBuffer(), which requires the new data to fit the original allocation), then queues a
    // staging upload into it. Preserves the registry name, so existing fg.buffer(name) bindings keep
    // resolving. Like replaceUploadedImage(), the old buffer is retired rather than destroyed.
    void replaceUploadedBuffer(const std::string &name, const void *data, VkDeviceSize size, VkBufferUsageFlags usage);

    // Reallocates a dynamic buffer (all its per-frame copies) while preserving its registry name; the
    // old allocations are retired, and the contents start zeroed.
    void replaceDynamicBuffer(const std::string &name, VkDeviceSize size, VkBufferUsageFlags usage);

    // For a dynamic buffer, the copy belonging to the current frame (see frameSlot()).
    AllocatedBuffer       *getBuffer(const std::string &name);
    const AllocatedBuffer *getBuffer(const std::string &name) const;
    // The copy a given frame slot uses (static buffers have one, whatever the slot).
    const AllocatedBuffer *getBuffer(const std::string &name, uint32_t frameSlot) const;
    bool                   hasBuffer(const std::string &name) const;
    // True for dynamic buffers with more than one copy, i.e. ones whose descriptors differ per frame.
    bool isPerFrame(const std::string &name) const;
    bool isDynamic(const std::string &name) const;

    // Copies a buffer's contents back to the CPU, as of the last submitted frame (pending uploads
    // are flushed first). Waits for the device to go idle, so it's for tests, debugging and
    // occasional readback, not per-frame use. Every registry buffer can be read back; for a dynamic
    // buffer this is simply its latest CPU-written contents.
    std::vector<std::byte> readBuffer(const std::string &name);

    // -----------------------------------------------------------------------
    // Replacement tracking and deferred destruction
    // -----------------------------------------------------------------------

    // Bumped each time a resource is replaced under the same name (a new VkBuffer/VkImage), so
    // anything that captured the old handle — e.g. descriptor sets written at compile — can tell
    // it needs refreshing. 0 for resources never replaced.
    uint64_t generation(const std::string &name) const;

    // Called by the frame loop once frame `frame` may start recording: every frame up to and
    // including `lastCompletedFrame` has finished on the GPU, so resources retired during those
    // frames are destroyed now. Frames are numbered from 1; 0 means "before the first frame".
    void beginFrame(uint64_t frame, uint64_t lastCompletedFrame);

    // How many frames the frame loop keeps in flight; dynamic buffers get that many copies. Set by
    // the owner (Viewer) before any dynamic buffer is registered. Defaults to 1 (no overlap).
    void     setFramesInFlight(uint32_t count);
    uint32_t framesInFlight() const { return m_framesInFlight; }
    // Which per-frame copy the current frame uses: frame % framesInFlight. The copy for frame F was
    // last used by frame F - framesInFlight, which has completed by the time F begins.
    uint32_t frameSlot() const { return static_cast<uint32_t>(m_currentFrame % m_framesInFlight); }

    // -----------------------------------------------------------------------
    // Upload queue — called automatically by FrameGraph::execute()
    // -----------------------------------------------------------------------

    bool hasPendingUploads() const { return !m_pendingUploads.empty(); }
    void flushUploads();

private:
    struct ImageEntry
    {
        AllocatedImage     image;
        VkFormat           format;
        VkImageUsageFlags  usage;
        VkImageAspectFlags aspect;
        ExtentSpec         extentSpec = ExtentSpec::swapchain();
        VkExtent2D         extent;
        bool               persistent = false;
        bool               external   = false;
        // Extended image properties — used by allocateImageEntry and rebuild
        uint32_t           mipLevels     = 1;
        uint32_t           arrayLayers   = 1;
        VkImageCreateFlags imageFlags    = 0;
        VkImageViewType    viewType      = VK_IMAGE_VIEW_TYPE_2D;
        bool               hasMipViews   = false;
        VkImageLayout      initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout      currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    struct BufferEntry
    {
        AllocatedBuffer    buffer; // static buffers; dynamic ones use frameCopies
        VkDeviceSize       size;
        VkBufferUsageFlags usage;
        VmaMemoryUsage     memoryUsage;

        // Dynamic buffers: one persistently mapped copy per frame in flight, the latest CPU-written
        // contents, and which version of them each copy holds.
        std::vector<AllocatedBuffer> frameCopies;
        std::vector<std::byte>       contents;
        uint64_t                     version = 0;
        std::vector<uint64_t>        copyVersions;

        bool isDynamic() const { return memoryUsage == VMA_MEMORY_USAGE_CPU_TO_GPU; }
    };

    struct PendingUpload
    {
        AllocatedBuffer staging;
        std::string     resourceName;
        enum class Type
        {
            Buffer,
            Image
        } type;
    };

    enum class ImageUploadMode
    {
        Create,
        Replace,
    };

    // A replaced resource the GPU may still be using, destroyed by beginFrame() once the frame it
    // was retired in has completed.
    struct RetiredResource
    {
        std::variant<AllocatedBuffer, AllocatedImage> resource;
        uint64_t                                      retiredInFrame;
    };

    void retire(AllocatedBuffer buffer);
    void retire(AllocatedImage image);

    // Creates a dynamic buffer's per-frame copies, zero-filled.
    BufferEntry createDynamicEntry(const std::string &name, VkDeviceSize size, VkBufferUsageFlags usage);
    // Writes the entry's latest contents into the given copy.
    void syncCopy(BufferEntry &entry, uint32_t slot);
    void destroyRetired(RetiredResource &retired);

    // Records with `record`, submits on the graphics queue and waits for completion.
    void runOneShotCommands(const std::function<void(VkCommandBuffer)> &record);

    void allocateImageEntry(const std::string &name, ImageEntry &entry);
    void queueImageUpload(const std::string &name, const void *data, uint32_t width, uint32_t height, VkFormat format,
                          bool generateMipmaps, ImageUploadMode mode);
    void initCommandPool();

    void setDebugName(VkObjectType objectType, uint64_t objectHandle, const std::string &name);

    const VulkanContext &m_ctx;
    Allocator           &m_allocator;
    VkExtent2D           m_defaultExtent;

    std::unordered_map<std::string, ImageEntry>               m_images;
    std::unordered_map<std::string, std::vector<std::string>> m_imageArrays;
    std::unordered_map<std::string, BufferEntry>              m_buffers;

    std::vector<PendingUpload> m_pendingUploads;

    std::vector<RetiredResource>              m_retired;
    std::unordered_map<std::string, uint64_t> m_generations;
    uint64_t                                  m_currentFrame   = 0;
    uint32_t                                  m_framesInFlight = 1;

    VkCommandPool m_uploadPool = VK_NULL_HANDLE;
};

} // namespace lr
