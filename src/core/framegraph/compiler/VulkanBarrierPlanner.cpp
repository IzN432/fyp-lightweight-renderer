#include "VulkanBarrierPlanner.hpp"

#include "core/vulkan/VkFormatUtils.hpp"

#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>

VkPipelineStageFlags2 lr::framegraph::stagesForShader(VkShaderStageFlags stages)
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

namespace
{

using lr::AccessMode;
using lr::BufferUsage;
using lr::BufferUse;
using lr::ImageUsage;
using lr::ImageUse;
using lr::framegraph::BarrierResourceKind;
using lr::framegraph::stagesForShader;
using lr::framegraph::VulkanResourceState;

struct RequiredAccess
{
    BarrierResourceKind kind;
    VulkanResourceState state;
    bool                writes = false;
    // Multisample images only: the pass needs contents an earlier pass rendered this frame.
    bool loads = false;
};

RequiredAccess accessForImage(const ImageUse &use)
{
    const bool reads  = use.access != AccessMode::Write;
    const bool writes = use.access != AccessMode::Read;
    switch (use.usage)
    {
        case ImageUsage::Sampled:
        case ImageUsage::SampledMultisample:
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
            if (use.isReadOnlyDepth())
            {
                return {BarrierResourceKind::Image,
                        {VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                         VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL}};
            }
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
    if (incoming.kind != BarrierResourceKind::Buffer && existing.state.layout != incoming.state.layout)
    {
        throw std::runtime_error("FrameGraph: pass uses image '" + name + "' with incompatible usages");
    }
    existing.state.stages |= incoming.state.stages;
    existing.state.access |= incoming.state.access;
    existing.writes |= incoming.writes;
    existing.loads |= incoming.loads;
}

using MultisampleFormats = std::map<std::string, VkFormat>;

// Formats of the attachments that MSAA passes render, i.e. of every private multisample image.
MultisampleFormats collectMultisampleFormats(std::span<const lr::PassDesc>     passes,
                                             const lr::ResourceHandleRegistry &resources)
{
    MultisampleFormats formats;
    for (const lr::PassDesc &pass : passes)
    {
        if (pass.graphics.samples == VK_SAMPLE_COUNT_1_BIT)
        {
            continue;
        }
        for (const ImageUse &use : pass.imageUses)
        {
            if (use.isAttachment())
            {
                formats.emplace(resources.name(use.image), use.format);
            }
        }
    }
    return formats;
}

std::map<ResourceKey, RequiredAccess> collectPassAccesses(const lr::PassDesc               &pass,
                                                          const lr::ResourceHandleRegistry &resources,
                                                          const MultisampleFormats         &multisampleFormats)
{
    const bool                            multisampled = pass.graphics.samples != VK_SAMPLE_COUNT_1_BIT;
    std::map<ResourceKey, RequiredAccess> accesses;
    for (const ImageUse &use : pass.imageUses)
    {
        const std::string &name   = resources.name(use.image);
        RequiredAccess     access = accessForImage(use);

        if (use.usage == ImageUsage::SampledMultisample)
        {
            // sampler2DMS reads the private image, never the public resolve destination.
            const auto format = multisampleFormats.find(name);
            if (format == multisampleFormats.end())
            {
                throw std::runtime_error("FrameGraph: pass '" + pass.name + "' samples unresolved image '" + name +
                                         "' without an MSAA producer");
            }
            access.kind  = BarrierResourceKind::MultisampleImage;
            access.loads = true;
            if (lr::isDepthFormat(format->second))
            {
                access.state.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            }
            mergeAccess(accesses, name, access);
            continue;
        }
        if (!multisampled || !use.isAttachment())
        {
            mergeAccess(accesses, name, access);
            continue;
        }

        // An MSAA pass renders into the private image...
        RequiredAccess target = access;
        target.kind           = BarrierResourceKind::MultisampleImage;
        target.loads          = use.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
        if (use.isReadOnlyDepth())
        {
            mergeAccess(accesses, name, target);
            continue;
        }

        // ...and resolves it into the public image at vkCmdEndRendering. The spec places resolves in
        // COLOR_ATTACHMENT_OUTPUT with color-attachment access for depth attachments too; depth also keeps
        // its fragment-test scope so the resolve destination is covered however the driver models it.
        target.state.stages |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        target.state.access |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
        mergeAccess(accesses, name, target);

        const bool     depth = use.usage == ImageUsage::DepthAttachment;
        RequiredAccess resolve{BarrierResourceKind::Image,
                               {VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                access.state.layout},
                               true};
        if (depth)
        {
            resolve.state.stages |=
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            resolve.state.access |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        }
        mergeAccess(accesses, name, resolve);
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
        // Accesses since the last barrier: the source of the next one.
        VulkanResourceState state;
        bool                writes = false;
        // What produced the current contents (the last write or layout transition), as a barrier source.
        // A read the last barrier's destination didn't cover (another stage or access type) still has to
        // wait for it. Unset until the frame writes or transitions the resource.
        std::optional<VulkanResourceState> producer;
    };
    std::map<ResourceKey, TrackedState> states;
    VulkanBarrierPlan                   plan;
    plan.beforePass.resize(passes.size());
    const MultisampleFormats multisampleFormats = collectMultisampleFormats(passes, resources);

    for (size_t passIndex : sortedPassIndices)
    {
        if (passIndex >= passes.size())
        {
            throw std::out_of_range("FrameGraph: barrier plan contains invalid pass index");
        }
        for (const auto &[key, required] : collectPassAccesses(passes[passIndex], resources, multisampleFormats))
        {
            auto stateIt = states.find(key);
            if (stateIt == states.end())
            {
                // Every resource is reused by successive submissions (the previous frame may still be on
                // the GPU), and the compiled barriers are recorded unchanged every frame. A barrier at the
                // first use therefore waits for all earlier queue work, which also chains it after the
                // swapchain-acquire semaphore wait. Read-only first uses emit no barrier, so this costs
                // nothing for resources a frame never writes or transitions.
                VulkanResourceState initial{.stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                            .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
                if (key.kind == BarrierResourceKind::MultisampleImage)
                {
                    // Private images keep nothing between frames, so each frame starts from UNDEFINED.
                    if (required.loads)
                    {
                        throw std::runtime_error("FrameGraph: pass '" + passes[passIndex].name +
                                                 "' reads the multisample contents of '" + key.name +
                                                 "' before any MSAA pass renders them this frame");
                    }
                } else if (key.kind == BarrierResourceKind::Image)
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
                key.kind != BarrierResourceKind::Buffer && current.state.layout != required.state.layout;
            if (layoutChanged || current.writes || required.writes)
            {
                plan.beforePass[passIndex].push_back({key.kind, key.name, current.state, required.state});
                std::optional<VulkanResourceState> producer = current.producer;
                if (current.writes || layoutChanged)
                {
                    // Waiting on the writer's stages and on this barrier's destination stages chains a later
                    // reader after both the write and the layout transition.
                    producer = VulkanResourceState{.stages = current.state.stages | required.state.stages,
                                                   .access = current.writes ? current.state.access : VK_ACCESS_2_NONE,
                                                   .layout = required.state.layout};
                }
                current = {.state = required.state, .writes = required.writes, .producer = producer};
            } else
            {
                // Same layout, no writes: only a reader the earlier barrier didn't make the contents visible
                // to needs one — e.g. a depth test after a shader sampled the same depth.
                const bool uncovered = (required.state.stages & ~current.state.stages) != 0 ||
                                       (required.state.access & ~current.state.access) != 0;
                if (current.producer && uncovered)
                {
                    plan.beforePass[passIndex].push_back({key.kind, key.name, *current.producer, required.state});
                }
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
            // ALL_COMMANDS rather than NONE: the transition must happen-before whatever follows the graph
            // (e.g. the submit's signal semaphore that presentation waits on), and an empty destination
            // scope would not chain into it.
            const VulkanResourceState destination{
                .stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .access = VK_ACCESS_2_NONE, .layout = requiredLayout};
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
