#include "ImageReadback.hpp"
#include "core/vulkan/VkResultUtils.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace lr
{

ImageReadback::ImageReadback(const VulkanContext &ctx, Allocator &alloc) : m_ctx(ctx), m_alloc(alloc)
{
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = static_cast<uint32_t>(m_ctx.getGraphicsQueueFamily());
    checkVk(vkCreateCommandPool(m_ctx.getDevice(), &ci, nullptr, &m_commandPool), "ImageReadback: vkCreateCommandPool");

    try
    {
        m_staging = m_alloc.createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
        m_stagingPixels = 1;
    } catch (...)
    {
        vkDestroyCommandPool(m_ctx.getDevice(), m_commandPool, nullptr);
        m_commandPool = VK_NULL_HANDLE;
        throw;
    }
}

ImageReadback::~ImageReadback()
{
    if (m_commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(m_ctx.getDevice(), m_commandPool, nullptr);
    }
    m_alloc.destroy(m_staging);
}

void ImageReadback::ensureStagingCapacity(uint32_t pixelCount)
{
    if (pixelCount <= m_stagingPixels)
    {
        return;
    }
    AllocatedBuffer replacement = m_alloc.createBuffer(static_cast<VkDeviceSize>(pixelCount) * sizeof(uint32_t),
                                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    m_alloc.destroy(m_staging);
    m_staging       = replacement;
    m_stagingPixels = pixelCount;
}

std::vector<uint32_t> ImageReadback::readRect(const ResourceRegistry &resources, const std::string &imageName,
                                              uint32_t x, uint32_t y, uint32_t width, uint32_t height)
{
    const AllocatedImage *img = resources.getImage(imageName);
    if (!img)
    {
        return {};
    }

    if (width == 0 || height == 0)
    {
        return {};
    }
    if (x >= img->extent.width || y >= img->extent.height || width > img->extent.width - x ||
        height > img->extent.height - y)
    {
        throw std::out_of_range("ImageReadback: rectangle is outside the source image");
    }
    if (width > std::numeric_limits<uint32_t>::max() / height)
    {
        throw std::overflow_error("ImageReadback: rectangle pixel count overflows uint32_t");
    }

    const uint32_t pixelCount = width * height;
    ensureStagingCapacity(pixelCount);

    VkCommandBuffer             cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = m_commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    checkVk(vkAllocateCommandBuffers(m_ctx.getDevice(), &ai, &cmd), "ImageReadback: vkAllocateCommandBuffers");

    VkFence fence     = VK_NULL_HANDLE;
    bool    submitted = false;
    try
    {

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        checkVk(vkBeginCommandBuffer(cmd, &bi), "ImageReadback: vkBeginCommandBuffer");

        // Whichever layout the last frame left the image in — a pass that samples it (for example
        // OutlinePass reading the picking IDs) ends the frame in SHADER_READ_ONLY_OPTIMAL rather than
        // COLOR_ATTACHMENT_OPTIMAL. A barrier that names the wrong old layout is invalid, so take it
        // from the registry, which records what each frame leaves behind (see
        // CompiledFrameGraph::execute). ALL_COMMANDS covers whichever pass that last writer was.
        const VkImageLayout sourceLayout = resources.getImageLayout(imageName);

        VkImageMemoryBarrier2 toSrc{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        toSrc.srcStageMask     = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        toSrc.srcAccessMask    = VK_ACCESS_2_MEMORY_WRITE_BIT;
        toSrc.dstStageMask     = VK_PIPELINE_STAGE_2_COPY_BIT;
        toSrc.dstAccessMask    = VK_ACCESS_2_TRANSFER_READ_BIT;
        toSrc.oldLayout        = sourceLayout;
        toSrc.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.image            = img->image;
        toSrc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &toSrc;
        vkCmdPipelineBarrier2(cmd, &dep);

        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset      = {static_cast<int32_t>(x), static_cast<int32_t>(y), 0};
        region.imageExtent      = {width, height, 1};
        vkCmdCopyImageToBuffer(cmd, img->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_staging.buffer, 1, &region);

        // Hand the image back in the layout it arrived in, so the registry's record stays true and the
        // next frame's barriers still describe reality. A source layout of UNDEFINED has nothing to
        // restore (and is not a legal newLayout): the graph's first barrier for such an image declares
        // UNDEFINED as its old layout anyway, which accepts whatever layout the image is in.
        if (sourceLayout != VK_IMAGE_LAYOUT_UNDEFINED)
        {
            VkImageMemoryBarrier2 restore = toSrc;
            restore.srcStageMask          = VK_PIPELINE_STAGE_2_COPY_BIT;
            restore.srcAccessMask         = VK_ACCESS_2_TRANSFER_READ_BIT;
            restore.dstStageMask          = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            restore.dstAccessMask         = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            restore.oldLayout             = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            restore.newLayout             = sourceLayout;
            dep.pImageMemoryBarriers      = &restore;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        checkVk(vkEndCommandBuffer(cmd), "ImageReadback: vkEndCommandBuffer");

        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        checkVk(vkCreateFence(m_ctx.getDevice(), &fi, nullptr, &fence), "ImageReadback: vkCreateFence");

        VkCommandBufferSubmitInfo cbInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cbInfo.commandBuffer = cmd;
        VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos    = &cbInfo;
        checkVk(vkQueueSubmit2(m_ctx.getGraphicsQueue(), 1, &submit, fence), "ImageReadback: vkQueueSubmit2");
        submitted = true;
        checkVk(vkWaitForFences(m_ctx.getDevice(), 1, &fence, VK_TRUE, UINT64_MAX), "ImageReadback: vkWaitForFences");
    } catch (...)
    {
        if (submitted)
        {
            (void)vkDeviceWaitIdle(m_ctx.getDevice());
        }
        if (fence != VK_NULL_HANDLE)
        {
            vkDestroyFence(m_ctx.getDevice(), fence, nullptr);
        }
        vkFreeCommandBuffers(m_ctx.getDevice(), m_commandPool, 1, &cmd);
        throw;
    }
    vkDestroyFence(m_ctx.getDevice(), fence, nullptr);
    vkFreeCommandBuffers(m_ctx.getDevice(), m_commandPool, 1, &cmd);

    std::vector<uint32_t> result(pixelCount);
    std::memcpy(result.data(), m_staging.info.pMappedData, pixelCount * sizeof(uint32_t));
    return result;
}

uint32_t ImageReadback::readPixel(const ResourceRegistry &resources, const std::string &imageName, uint32_t x,
                                  uint32_t y)
{
    auto pixels = readRect(resources, imageName, x, y, 1, 1);
    return pixels.empty() ? kNoData : pixels[0];
}

} // namespace lr
