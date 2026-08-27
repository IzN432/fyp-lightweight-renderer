#include "VulkanBarrierPlanner.hpp"

#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace
{

using lr::BindingAccess;
using lr::BindingDesc;
using lr::PassDesc;
using lr::framegraph::BarrierResourceKind;
using lr::framegraph::VulkanResourceState;

bool isDepthFormat(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D24_UNORM_S8_UINT ||
           format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

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
    bool writes = false;
};

std::optional<RequiredAccess> accessForBinding(const BindingDesc &binding)
{
    const VkPipelineStageFlags2 stages = stagesForShader(binding.stages);
    const bool reads = binding.access != BindingAccess::Write;
    const bool writes = binding.access != BindingAccess::Read;

    switch (binding.type)
    {
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            return RequiredAccess{
                .kind = BarrierResourceKind::Image,
                .state = {stages, VK_ACCESS_2_SHADER_READ_BIT, binding.imageLayout},
            };
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            return RequiredAccess{
                .kind = BarrierResourceKind::Image,
                .state = {VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                          VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT, binding.imageLayout},
            };
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
            VkAccessFlags2 access = VK_ACCESS_2_NONE;
            if (reads)
            {
                access |= VK_ACCESS_2_SHADER_READ_BIT;
            }
            if (writes)
            {
                access |= VK_ACCESS_2_SHADER_WRITE_BIT;
            }
            return RequiredAccess{
                .kind = BarrierResourceKind::Image,
                .state = {stages, access, VK_IMAGE_LAYOUT_GENERAL},
                .writes = writes,
            };
        }
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            return RequiredAccess{
                .kind = BarrierResourceKind::Buffer,
                .state = {stages, VK_ACCESS_2_UNIFORM_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED},
            };
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
            VkAccessFlags2 access = VK_ACCESS_2_NONE;
            if (reads)
            {
                access |= VK_ACCESS_2_SHADER_READ_BIT;
            }
            if (writes)
            {
                access |= VK_ACCESS_2_SHADER_WRITE_BIT;
            }
            return RequiredAccess{
                .kind = BarrierResourceKind::Buffer,
                .state = {stages, access, VK_IMAGE_LAYOUT_UNDEFINED},
                .writes = writes,
            };
        }
        default:
            return std::nullopt;
    }
}

RequiredAccess accessForAttachment(const lr::ResourceDesc &attachment)
{
    const bool loads = attachment.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
    if (isDepthFormat(attachment.format))
    {
        VkAccessFlags2 access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        if (loads)
        {
            access |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        }
        return {
            .kind = BarrierResourceKind::Image,
            .state = {VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                          VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                      access, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
            .writes = true,
        };
    }

    VkAccessFlags2 access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    if (loads)
    {
        access |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
    }
    return {
        .kind = BarrierResourceKind::Image,
        .state = {VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, access,
                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        .writes = true,
    };
}

struct ResourceKey
{
    BarrierResourceKind kind;
    std::string name;

    friend bool operator<(const ResourceKey &lhs, const ResourceKey &rhs)
    {
        return std::tie(lhs.kind, lhs.name) < std::tie(rhs.kind, rhs.name);
    }
};

void mergeAccess(std::map<ResourceKey, RequiredAccess> &accesses,
                 const std::string &name,
                 RequiredAccess incoming)
{
    ResourceKey key{incoming.kind, name};
    auto [it, inserted] = accesses.emplace(key, incoming);
    if (inserted)
    {
        return;
    }

    RequiredAccess &existing = it->second;
    if (incoming.kind == BarrierResourceKind::Image &&
        existing.state.layout != incoming.state.layout)
    {
        throw std::runtime_error("FrameGraph: pass uses image '" + name +
                                 "' with incompatible layouts");
    }

    existing.state.stages |= incoming.state.stages;
    existing.state.access |= incoming.state.access;
    existing.writes |= incoming.writes;
}

std::map<ResourceKey, RequiredAccess> collectPassAccesses(const PassDesc &pass)
{
    std::map<ResourceKey, RequiredAccess> accesses;
    for (const BindingDesc &binding : pass.bindings)
    {
        if (auto access = accessForBinding(binding))
        {
            mergeAccess(accesses, binding.resourceName, *access);
        }
    }
    for (const lr::ResourceDesc &attachment : pass.writes)
    {
        mergeAccess(accesses, attachment.name, accessForAttachment(attachment));
    }
    for (const PassDesc::VertexBufferRef &vertexBuffer : pass.vertexBufferRefs)
    {
        mergeAccess(accesses, vertexBuffer.bufferName, {
            .kind = BarrierResourceKind::Buffer,
            .state = {VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
                      VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED},
        });
    }
    if (!pass.indexBufferName.empty())
    {
        mergeAccess(accesses, pass.indexBufferName, {
            .kind = BarrierResourceKind::Buffer,
            .state = {VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
                      VK_ACCESS_2_INDEX_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED},
        });
    }
    return accesses;
}

} // namespace

namespace lr::framegraph
{

VulkanBarrierPlan planVulkanBarriers(
    std::span<const PassDesc> passes,
    std::span<const size_t> sortedPassIndices,
    const std::unordered_map<std::string, VkImageLayout> &initialImageLayouts)
{
    struct TrackedState
    {
        VulkanResourceState state;
        bool writes = false;
    };

    std::map<ResourceKey, TrackedState> states;
    VulkanBarrierPlan plan;
    plan.beforePass.resize(passes.size());

    for (size_t passIndex : sortedPassIndices)
    {
        if (passIndex >= passes.size())
        {
            throw std::out_of_range("FrameGraph: barrier plan contains invalid pass index");
        }

        for (const auto &[key, required] : collectPassAccesses(passes[passIndex]))
        {
            auto stateIt = states.find(key);
            if (stateIt == states.end())
            {
                VulkanResourceState initial{};
                if (key.kind == BarrierResourceKind::Image)
                {
                    const auto layout = initialImageLayouts.find(key.name);
                    if (layout != initialImageLayouts.end())
                    {
                        initial.layout = layout->second;
                    }
                }
                stateIt = states.emplace(key, TrackedState{.state = initial}).first;
            }

            TrackedState &current = stateIt->second;
            const bool layoutChanged = key.kind == BarrierResourceKind::Image &&
                                       current.state.layout != required.state.layout;
            const bool needsBarrier = layoutChanged || current.writes || required.writes;

            if (needsBarrier)
            {
                plan.beforePass[passIndex].push_back({
                    .kind = key.kind,
                    .resourceName = key.name,
                    .source = current.state,
                    .destination = required.state,
                });
                current = {.state = required.state, .writes = required.writes};
            }
            else
            {
                // Consecutive reads need no barrier. Accumulate all reader stages
                // so a later write waits for every outstanding read domain.
                current.state.stages |= required.state.stages;
                current.state.access |= required.state.access;
                current.state.layout = required.state.layout;
            }
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
