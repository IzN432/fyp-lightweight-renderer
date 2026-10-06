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
    // The private multisample image behind an attachment name, shared by every MSAA pass rendering to
    // it and read by sampler2DMS bindings. Its contents never carry over between frames.
    MultisampleImage,
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

// Shader stages that touch a resource -> the pipeline stages a barrier must name for them.
VkPipelineStageFlags2 stagesForShader(VkShaderStageFlags stages);

// Converts the current Vulkan-facing PassDesc frontend into synchronization
// barriers. Resource ordering is supplied by GraphCompiler's execution plan.
// Attachments of an MSAA pass are planned twice: the private multisample image as the render target,
// and the public image as the resolve destination (skipped for read-only depth, which never resolves).
VulkanBarrierPlan planVulkanBarriers(std::span<const PassDesc> passes, const ResourceHandleRegistry &resources,
                                     std::span<const size_t>                               sortedPassIndices,
                                     const std::unordered_map<std::string, VkImageLayout> &initialImageLayouts  = {},
                                     const std::unordered_map<std::string, VkImageLayout> &requiredFinalLayouts = {});

} // namespace lr::framegraph
