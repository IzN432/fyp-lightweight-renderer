#include "ImageReadback.hpp"

#include <cstring>

namespace lr
{

ImageReadback::ImageReadback(const VulkanContext &ctx, Allocator &alloc)
    : m_ctx(ctx), m_alloc(alloc)
{
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                          VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = static_cast<uint32_t>(m_ctx.getGraphicsQueueFamily());
    vkCreateCommandPool(m_ctx.getDevice(), &ci, nullptr, &m_commandPool);

    m_staging      = m_alloc.createBuffer(sizeof(uint32_t),
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          VMA_MEMORY_USAGE_CPU_ONLY);
    m_stagingPixels = 1;
}

ImageReadback::~ImageReadback()
{
    vkDestroyCommandPool(m_ctx.getDevice(), m_commandPool, nullptr);
    m_alloc.destroy(m_staging);
}

void ImageReadback::ensureStagingCapacity(uint32_t pixelCount)
{
    if (pixelCount <= m_stagingPixels)
        return;
    m_alloc.destroy(m_staging);
    m_staging       = m_alloc.createBuffer(pixelCount * sizeof(uint32_t),
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VMA_MEMORY_USAGE_CPU_ONLY);
    m_stagingPixels = pixelCount;
}

std::vector<uint32_t> ImageReadback::readRect(const ResourceRegistry &resources,
                                               const std::string &imageName,
                                               uint32_t x, uint32_t y,
                                               uint32_t width, uint32_t height)
{
    const AllocatedImage *img = resources.getImage(imageName);
    if (!img)
        return {};

    const uint32_t pixelCount = width * height;
    ensureStagingCapacity(pixelCount);

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = m_commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(m_ctx.getDevice(), &ai, &cmd);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    VkImageMemoryBarrier2 toSrc{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    toSrc.srcStageMask       = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toSrc.srcAccessMask      = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toSrc.dstStageMask       = VK_PIPELINE_STAGE_2_COPY_BIT;
    toSrc.dstAccessMask      = VK_ACCESS_2_TRANSFER_READ_BIT;
    toSrc.oldLayout          = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toSrc.newLayout          = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSrc.image              = img->image;
    toSrc.subresourceRange   = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &toSrc;
    vkCmdPipelineBarrier2(cmd, &dep);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset      = {static_cast<int32_t>(x), static_cast<int32_t>(y), 0};
    region.imageExtent      = {width, height, 1};
    vkCmdCopyImageToBuffer(cmd, img->image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           m_staging.buffer, 1, &region);

    VkImageMemoryBarrier2 toAttach  = toSrc;
    toAttach.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    toAttach.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    toAttach.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toAttach.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toAttach.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toAttach.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    dep.pImageMemoryBarriers = &toAttach;
    vkCmdPipelineBarrier2(cmd, &dep);

    vkEndCommandBuffer(cmd);

    VkFence fence;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(m_ctx.getDevice(), &fi, nullptr, &fence);

    VkCommandBufferSubmitInfo cbInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbInfo.commandBuffer = cmd;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos    = &cbInfo;
    vkQueueSubmit2(m_ctx.getGraphicsQueue(), 1, &submit, fence);
    vkWaitForFences(m_ctx.getDevice(), 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(m_ctx.getDevice(), fence, nullptr);
    vkFreeCommandBuffers(m_ctx.getDevice(), m_commandPool, 1, &cmd);

    std::vector<uint32_t> result(pixelCount);
    std::memcpy(result.data(), m_staging.info.pMappedData, pixelCount * sizeof(uint32_t));
    return result;
}

uint32_t ImageReadback::readPixel(const ResourceRegistry &resources,
                                   const std::string &imageName,
                                   uint32_t x, uint32_t y)
{
    auto pixels = readRect(resources, imageName, x, y, 1, 1);
    return pixels.empty() ? kNoData : pixels[0];
}

}  // namespace lr
