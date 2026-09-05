#pragma once

#include "core/framegraph/PassDefinition.hpp"

#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace lr::framegraph
{

enum class BarrierResourceKind
{
    Image,
    Buffer,
};

struct VulkanResourceState
{
    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2        access = VK_ACCESS_2_NONE;
    VkImageLayout         layout = VK_IMAGE_LAYOUT_UNDEFINED; // ignored for buffers
};

struct PlannedBarrier
{
    BarrierResourceKind kind = BarrierResourceKind::Image;
    std::string         resourceName;
    VulkanResourceState source;
    VulkanResourceState destination;
};

struct VulkanBarrierPlan
{
    // Indexed by pass declaration index, regardless of execution order.
    std::vector<std::vector<PlannedBarrier>>       beforePass;
    std::vector<PlannedBarrier>                    afterGraph;
    std::unordered_map<std::string, VkImageLayout> finalImageLayouts;
};

// Converts the current Vulkan-facing PassDesc frontend into synchronization
// barriers. Resource ordering is supplied by GraphCompiler's execution plan.
VulkanBarrierPlan planVulkanBarriers(std::span<const PassDesc> passes, const ResourceHandleRegistry &resources,
                                     std::span<const size_t>                               sortedPassIndices,
                                     const std::unordered_map<std::string, VkImageLayout> &initialImageLayouts  = {},
                                     const std::unordered_map<std::string, VkImageLayout> &requiredFinalLayouts = {});

} // namespace lr::framegraph
