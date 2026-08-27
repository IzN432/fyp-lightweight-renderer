#pragma once

#include "PassBuilder.hpp"

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
    VkExtent2D         extent{};
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

PassDesc &appendPass(std::vector<PassDesc> &passes, std::string name);

// These functions intentionally reproduce today's behavior. They form the
// characterization seam that later frame-graph changes will be tested against.
std::vector<size_t> sortPasses(std::span<const PassDesc> passes);

std::vector<PlannedImage> planAttachmentImages(
    std::span<const PassDesc> passes,
    VkExtent2D defaultExtent,
    const std::unordered_set<std::string> &existingImages = {});

// Stable, handle-free text suitable for logs and snapshot comparisons.
std::string dumpTopology(
    std::span<const PassDesc> passes,
    std::span<const size_t> sortedPassIndices,
    std::span<const std::vector<BarrierDebugInfo>> barriersByPass = {});

} // namespace lr::framegraph
