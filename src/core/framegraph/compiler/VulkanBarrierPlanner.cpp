#include "VulkanBarrierPlanner.hpp"

#include <map>
#include <stdexcept>
#include <tuple>

namespace
{

using lr::AccessMode;
using lr::BufferUsage;
using lr::BufferUse;
using lr::ImageUsage;
using lr::ImageUse;
using lr::framegraph::BarrierResourceKind;
using lr::framegraph::VulkanResourceState;

VkPipelineStageFlags2 stagesForShader(VkShaderStageFlags stages)
{
    if (stages == VK_SHADER_STAGE_ALL)
    {
        return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
    if (stages == VK_SHADER_STAGE_ALL_GRAPHICS)
    {
        return VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
    }
    VkPipelineStageFlags2 result = VK_PIPELINE_STAGE_2_NONE;
    if (stages & VK_SHADER_STAGE_VERTEX_BIT)
    {
        result |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)
    {
        result |= VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)
    {
        result |= VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_GEOMETRY_BIT)
    {
        result |= VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_FRAGMENT_BIT)
    {
        result |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_COMPUTE_BIT)
    {
        result |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    }
    return result == VK_PIPELINE_STAGE_2_NONE ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : result;
}

struct RequiredAccess
{
    BarrierResourceKind kind;
    VulkanResourceState state;
    bool                writes = false;
};

RequiredAccess accessForImage(const ImageUse &use)
{
    const bool reads  = use.access != AccessMode::Write;
    const bool writes = use.access != AccessMode::Read;
    switch (use.usage)
    {
        case ImageUsage::Sampled:
        case ImageUsage::SampledArray:
            return {
                BarrierResourceKind::Image,
                {stagesForShader(use.stages), VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
        case ImageUsage::SampledDepth:
            return {BarrierResourceKind::Image,
                    {stagesForShader(use.stages), VK_ACCESS_2_SHADER_READ_BIT,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL}};
        case ImageUsage::Storage: {
            VkAccessFlags2 access = VK_ACCESS_2_NONE;
            if (reads)
            {
                access |= VK_ACCESS_2_SHADER_READ_BIT;
            }
            if (writes)
            {
                access |= VK_ACCESS_2_SHADER_WRITE_BIT;
            }
            return {BarrierResourceKind::Image, {stagesForShader(use.stages), access, VK_IMAGE_LAYOUT_GENERAL}, writes};
        }
        case ImageUsage::ColorAttachment: {
            VkAccessFlags2 access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            if (reads)
            {
                access |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
            }
            return {BarrierResourceKind::Image,
                    {VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, access, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
                    true};
        }
        case ImageUsage::DepthAttachment: {
            VkAccessFlags2 access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            if (reads)
            {
                access |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            }
            return {BarrierResourceKind::Image,
                    {VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, access,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
                    true};
        }
    }
    throw std::runtime_error("FrameGraph: unsupported image usage");
}

RequiredAccess accessForBuffer(const BufferUse &use)
{
    const bool reads  = use.access != AccessMode::Write;
    const bool writes = use.access != AccessMode::Read;
    switch (use.usage)
    {
        case BufferUsage::Uniform:
            return {BarrierResourceKind::Buffer,
                    {stagesForShader(use.stages), VK_ACCESS_2_UNIFORM_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED}};
        case BufferUsage::Storage: {
            VkAccessFlags2 access = VK_ACCESS_2_NONE;
            if (reads)
            {
                access |= VK_ACCESS_2_SHADER_READ_BIT;
            }
            if (writes)
            {
                access |= VK_ACCESS_2_SHADER_WRITE_BIT;
            }
            return {
                BarrierResourceKind::Buffer, {stagesForShader(use.stages), access, VK_IMAGE_LAYOUT_UNDEFINED}, writes};
        }
        case BufferUsage::Vertex:
            return {BarrierResourceKind::Buffer,
                    {VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED}};
        case BufferUsage::Index:
            return {BarrierResourceKind::Buffer,
                    {VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT, VK_ACCESS_2_INDEX_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED}};
        case BufferUsage::Indirect:
            return {BarrierResourceKind::Buffer,
                    {VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED}};
    }
    throw std::runtime_error("FrameGraph: unsupported buffer usage");
}

struct ResourceKey
{
    BarrierResourceKind kind;
    std::string         name;
    friend bool         operator<(const ResourceKey &lhs, const ResourceKey &rhs)
    {
        return std::tie(lhs.kind, lhs.name) < std::tie(rhs.kind, rhs.name);
    }
};

void mergeAccess(std::map<ResourceKey, RequiredAccess> &accesses, const std::string &name, RequiredAccess incoming)
{
    const ResourceKey key{incoming.kind, name};
    auto [it, inserted] = accesses.emplace(key, incoming);
    if (inserted)
    {
        return;
    }
    RequiredAccess &existing = it->second;
    if (incoming.kind == BarrierResourceKind::Image && existing.state.layout != incoming.state.layout)
    {
        throw std::runtime_error("FrameGraph: pass uses image '" + name + "' with incompatible usages");
    }
    existing.state.stages |= incoming.state.stages;
    existing.state.access |= incoming.state.access;
    existing.writes |= incoming.writes;
}

std::map<ResourceKey, RequiredAccess> collectPassAccesses(const lr::PassDesc               &pass,
                                                          const lr::ResourceHandleRegistry &resources)
{
    std::map<ResourceKey, RequiredAccess> accesses;
    for (const ImageUse &use : pass.imageUses)
    {
        mergeAccess(accesses, resources.name(use.image), accessForImage(use));
    }
    for (const BufferUse &use : pass.bufferUses)
    {
        mergeAccess(accesses, resources.name(use.buffer), accessForBuffer(use));
    }
    return accesses;
}

} // namespace

namespace lr::framegraph
{

VulkanBarrierPlan planVulkanBarriers(std::span<const PassDesc> passes, const ResourceHandleRegistry &resources,
                                     std::span<const size_t>                               sortedPassIndices,
                                     const std::unordered_map<std::string, VkImageLayout> &initialImageLayouts,
                                     const std::unordered_map<std::string, VkImageLayout> &requiredFinalLayouts)
{
    struct TrackedState
    {
        VulkanResourceState state;
        bool                writes = false;
    };
    std::map<ResourceKey, TrackedState> states;
    VulkanBarrierPlan                   plan;
    plan.beforePass.resize(passes.size());

    for (size_t passIndex : sortedPassIndices)
    {
        if (passIndex >= passes.size())
        {
            throw std::out_of_range("FrameGraph: barrier plan contains invalid pass index");
        }
        for (const auto &[key, required] : collectPassAccesses(passes[passIndex], resources))
        {
            auto stateIt = states.find(key);
            if (stateIt == states.end())
            {
                VulkanResourceState initial{};
                if (key.kind == BarrierResourceKind::Image)
                {
                    if (const auto it = initialImageLayouts.find(key.name); it != initialImageLayouts.end())
                    {
                        initial.layout = it->second;
                    }
                }
                stateIt = states.emplace(key, TrackedState{.state = initial}).first;
            }
            TrackedState &current = stateIt->second;
            const bool    layoutChanged =
                key.kind == BarrierResourceKind::Image && current.state.layout != required.state.layout;
            if (layoutChanged || current.writes || required.writes)
            {
                plan.beforePass[passIndex].push_back({key.kind, key.name, current.state, required.state});
                current = {.state = required.state, .writes = required.writes};
            } else
            {
                current.state.stages |= required.state.stages;
                current.state.access |= required.state.access;
                current.state.layout = required.state.layout;
            }
        }
    }
    for (const auto &[name, requiredLayout] : requiredFinalLayouts)
    {
        const ResourceKey key{BarrierResourceKind::Image, name};
        const auto        current = states.find(key);
        if (current == states.end())
        {
            throw std::runtime_error("FrameGraph: exported image '" + name + "' is not used by any pass");
        }
        if (current->second.state.layout != requiredLayout)
        {
            const VulkanResourceState destination{
                .stages = VK_PIPELINE_STAGE_2_NONE, .access = VK_ACCESS_2_NONE, .layout = requiredLayout};
            plan.afterGraph.push_back({BarrierResourceKind::Image, name, current->second.state, destination});
            current->second = {.state = destination, .writes = false};
        }
    }

    for (const auto &[key, tracked] : states)
    {
        if (key.kind == BarrierResourceKind::Image)
        {
            plan.finalImageLayouts.emplace(key.name, tracked.state.layout);
        }
    }
    return plan;
}

} // namespace lr::framegraph
