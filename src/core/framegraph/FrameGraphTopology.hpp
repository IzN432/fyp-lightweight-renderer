#pragma once

#include "PassDefinition.hpp"

#include <span>
#include <string>
#include <unordered_set>
#include <vector>

namespace lr::framegraph
{

struct PlannedImage
{
    std::string        name;
    VkFormat           format = VK_FORMAT_UNDEFINED;
    ExtentSpec         extent = ExtentSpec::swapchain();
    VkImageUsageFlags  usage  = 0;
    VkImageAspectFlags aspect = 0;
};

struct BarrierDebugInfo
{
    std::string           resourceName;
    VkPipelineStageFlags2 srcStage  = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2        srcAccess = VK_ACCESS_2_NONE;
    VkPipelineStageFlags2 dstStage  = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2        dstAccess = VK_ACCESS_2_NONE;
    VkImageLayout         oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout         newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
};

std::vector<size_t> sortPasses(std::span<const PassDesc> passes, const ResourceHandleRegistry &resources,
                               uint64_t passOwner);

std::vector<PlannedImage> planAttachmentImages(std::span<const PassDesc>              passes,
                                               const ResourceHandleRegistry          &resources,
                                               const std::unordered_set<std::string> &existingImages = {});

std::vector<VkExtent2D> planRenderingExtents(std::span<const PassDesc> passes, VkExtent2D defaultExtent);

// Stable, handle-free text suitable for logs and snapshot comparisons.
std::string dumpTopology(std::span<const PassDesc> passes, const ResourceHandleRegistry &resources,
                         std::span<const size_t>                        sortedPassIndices,
                         std::span<const std::vector<BarrierDebugInfo>> barriersByPass = {});

} // namespace lr::framegraph
