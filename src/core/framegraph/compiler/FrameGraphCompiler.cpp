#include "FrameGraphCompiler.hpp"

#include "GraphCompiler.hpp"
#include "ShaderInterface.hpp"
#include "VulkanBarrierPlanner.hpp"
#include "core/framegraph/CompiledFrameGraph.hpp"
#include "core/framegraph/FrameGraphTopology.hpp"
#include "core/framegraph/PassDescAdapter.hpp"
#include "core/pipeline/ComputePipeline.hpp"
#include "core/pipeline/GraphicsPipeline.hpp"
#include "core/vulkan/VkFormatUtils.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace lr
{

std::unique_ptr<CompiledFrameGraph> FrameGraphCompiler::compile(const FrameGraphDefinition &definition) const
{
    spdlog::info("FrameGraph: compiling {} passes...", definition.passes().size());

    // Shader/declaration mismatches are reported before any Vulkan object is created from them.
    for (const PassDesc &pass : definition.passes())
    {
        validateShaderInterface(pass);
    }

    // Validate and plan the backend-independent topology before creating any
    // Vulkan objects. The compiled graph then owns this independent snapshot.
    FrameGraphDefinition snapshot = definition;
    const auto           logicalGraph =
        framegraph::translatePassDescriptions(snapshot.passes(), snapshot.resources(), snapshot.owner());
    framegraph::ExecutionPlan executionPlan = framegraph::buildExecutionPlan(logicalGraph);

    auto result = std::unique_ptr<CompiledFrameGraph>(new CompiledFrameGraph(m_ctx, m_registry, std::move(snapshot)));
    result->m_executionPlan = std::move(executionPlan);
    result->m_passes.resize(result->m_definition.passes().size());

    result->m_sortedIndices.reserve(result->m_executionPlan.orderedPasses.size());
    for (framegraph::PassId pass : result->m_executionPlan.orderedPasses)
    {
        result->m_sortedIndices.push_back(pass.value);
    }
    allocateResources(*result);
    buildDescriptorSets(*result);
    buildPipelines(*result);
    buildBarriers(*result);

    if (spdlog::should_log(spdlog::level::debug))
    {
        spdlog::debug("{}", result->debugDump());
    }
    spdlog::info("FrameGraph: compiled OK");
    return result;
}

void FrameGraphCompiler::allocateResources(CompiledFrameGraph &graph) const
{
    const auto planned = framegraph::planAttachmentImages(graph.m_definition.passes(), graph.m_definition.resources());
    for (const auto &image : planned)
    {
        if (!m_registry.hasImage(image.name))
        {
            m_registry.registerImage(image.name, image.format, image.usage, image.extent, image.aspect);
        } else
        {
            m_registry.validateImage(image.name, image.format, image.usage, image.extent, image.aspect);
        }
    }

    const auto renderingExtents = framegraph::planRenderingExtents(graph.m_definition.passes(), m_registry.getExtent());
    for (size_t passIndex = 0; passIndex < renderingExtents.size(); ++passIndex)
    {
        graph.m_passes[passIndex].renderingExtent = renderingExtents[passIndex];
    }

    // Create private multisample render targets, one per attachment name. Public graph images stay
    // single-sampled and become resolve destinations, so ordinary sampler2D bindings work unchanged;
    // sampler2DMS bindings and later MSAA passes (e.g. one that LOADs the depth) select the private image.
    // planAttachmentImages already rejects conflicting formats and extents for one name.
    const auto passes = graph.m_definition.passes();
    for (size_t passIndex = 0; passIndex < passes.size(); ++passIndex)
    {
        const PassDesc &pass = passes[passIndex];
        if (pass.graphics.samples == VK_SAMPLE_COUNT_1_BIT)
        {
            continue;
        }
        for (const ImageUse &use : pass.imageUses)
        {
            if (!use.isAttachment())
            {
                continue;
            }
            const std::string &name = graph.m_definition.name(use.image);
            if (const auto existing = graph.m_multisampleImages.find(name); existing != graph.m_multisampleImages.end())
            {
                if (existing->second.samples != pass.graphics.samples)
                {
                    throw std::runtime_error("FrameGraph: pass '" + pass.name + "' renders attachment '" + name +
                                             "' with a different MSAA sample count than an earlier pass");
                }
                continue;
            }
            const bool               depth     = use.usage == ImageUsage::DepthAttachment;
            const VkSampleCountFlags supported = depth ? m_ctx.depthSampleCounts() : m_ctx.colorSampleCounts();
            if (!(supported & pass.graphics.samples))
            {
                throw std::runtime_error("FrameGraph: pass '" + pass.name +
                                         "' requests an unsupported MSAA "
                                         "sample count for one of its attachments");
            }
            ImageConfig config{};
            config.extent = {renderingExtents[passIndex].width, renderingExtents[passIndex].height, 1};
            config.format = use.format;
            config.usage = (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) |
                           VK_IMAGE_USAGE_SAMPLED_BIT;
            config.aspect  = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            config.samples = pass.graphics.samples;
            graph.m_multisampleImages.emplace(name, m_registry.allocator().createImage(config));
        }
    }

    for (const PassDesc &pass : graph.m_definition.passes())
    {
        for (const ImageUse &use : pass.imageUses)
        {
            if (use.isAttachment())
            {
                continue;
            }
            const VkImageUsageFlags requiredUsage =
                use.usage == ImageUsage::Storage ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT;
            m_registry.validateImageUsage(graph.m_definition.name(use.image), requiredUsage);
        }
    }
}

void FrameGraphCompiler::buildDescriptorSets(CompiledFrameGraph &graph) const
{
    const auto passes = graph.m_definition.passes();
    for (size_t index = 0; index < passes.size(); ++index)
    {
        const PassDesc &pass     = passes[index];
        auto           &compiled = graph.m_passes[index];

        if (pass.type == PassType::Custom)
        {
            continue;
        }

        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        std::unordered_set<uint32_t>              usedBindings;
        const auto addLayoutBinding = [&](uint32_t binding, VkDescriptorType type, uint32_t count,
                                          VkShaderStageFlags stages) {
            if (!usedBindings.insert(binding).second)
            {
                throw std::runtime_error("FrameGraph: duplicate descriptor binding " + std::to_string(binding) +
                                         " in pass '" + pass.name + "'");
            }
            layoutBindings.push_back(
                {.binding = binding, .descriptorType = type, .descriptorCount = count, .stageFlags = stages});
        };

        for (const ImageUse &use : pass.imageUses)
        {
            if (!use.isDescriptor())
            {
                continue;
            }
            const VkDescriptorType type = use.usage == ImageUsage::Storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                                           : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            addLayoutBinding(use.binding, type, use.descriptorCount, use.stages);
        }
        for (const BufferUse &use : pass.bufferUses)
        {
            if (!use.isDescriptor())
            {
                continue;
            }
            const std::string &name = graph.m_definition.name(use.buffer);
            if (use.access != AccessMode::Read && m_registry.isDynamic(name))
            {
                throw std::runtime_error("FrameGraph: pass '" + pass.name + "' writes dynamic buffer '" + name +
                                         "' — dynamic buffers are written by the CPU (one copy per frame in "
                                         "flight); use a static buffer for GPU writes");
            }
            const VkDescriptorType type = use.usage == BufferUsage::Uniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                                            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            addLayoutBinding(use.binding, type, 1, use.stages);
        }

        compiled.descriptorLayout = graph.m_descriptorAllocator.createLayout(layoutBindings);
        compiled.pipelineLayout   = graph.m_descriptorAllocator.createPipelineLayout(
            compiled.descriptorLayout, pass.pushConstantSize, pass.pushConstantStages);

        if (layoutBindings.empty())
        {
            continue;
        }

        // A pass binding a per-frame (dynamic) buffer gets one set per frame in flight, each pointing at
        // that frame slot's copy; everything else in the sets is identical.
        const bool     perFrame = std::ranges::any_of(pass.bufferUses, [&](const BufferUse &use) {
            return use.isDescriptor() && m_registry.isPerFrame(graph.m_definition.name(use.buffer));
        });
        const uint32_t setCount = perFrame ? m_registry.framesInFlight() : 1;
        for (uint32_t slot = 0; slot < setCount; ++slot)
        {
            const VkDescriptorSet descriptorSet = graph.m_descriptorAllocator.allocate(compiled.descriptorLayout);
            compiled.descriptorSets.push_back(descriptorSet);
            for (const ImageUse &use : pass.imageUses)
            {
                if (!use.isDescriptor())
                {
                    continue;
                }
                const std::string     &name    = graph.m_definition.name(use.image);
                const bool             storage = use.usage == ImageUsage::Storage;
                const VkDescriptorType type =
                    storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                const VkSampler     sampler = storage ? VK_NULL_HANDLE : graph.sampler(use.sampler);
                const VkImageLayout layout =
                    storage ? VK_IMAGE_LAYOUT_GENERAL
                            : (use.usage == ImageUsage::SampledDepth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                                     : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                if (use.usage == ImageUsage::SampledArray)
                {
                    const auto images = m_registry.getImageArray(name);
                    if (images.size() < use.descriptorCount)
                    {
                        throw std::runtime_error("FrameGraph: pass '" + pass.name + "' binds image array '" + name +
                                                 "' with too few slots");
                    }
                    std::vector<VkImageView> views;
                    views.reserve(use.descriptorCount);
                    for (uint32_t arrayIndex = 0; arrayIndex < use.descriptorCount; ++arrayIndex)
                    {
                        if (!images[arrayIndex])
                        {
                            throw std::runtime_error("FrameGraph: image array '" + name + "' has an empty slot");
                        }
                        views.push_back(images[arrayIndex]->view);
                    }
                    graph.m_descriptorAllocator.writeImageArray(descriptorSet, use.binding, views, sampler, layout,
                                                                type);
                    continue;
                }

                const bool            multisample = use.usage == ImageUsage::SampledMultisample;
                const AllocatedImage *image = multisample ? graph.multisampleImage(name) : m_registry.getImage(name);
                if (!image)
                {
                    throw std::runtime_error("FrameGraph: pass '" + pass.name + "' binds " +
                                             (multisample ? "an image with no MSAA producer '" : "unknown image '") +
                                             name + "'");
                }
                VkImageView view = image->view;
                if (use.boundMip != allImageMips)
                {
                    if (!storage)
                    {
                        throw std::runtime_error(
                            "FrameGraph: mip-specific views are only supported for storage images");
                    }
                    if (use.boundMip >= image->mipViews.size())
                    {
                        throw std::runtime_error("FrameGraph: pass '" + pass.name + "' requests invalid mip for '" +
                                                 name + "'");
                    }
                    view = image->mipViews[use.boundMip];
                }
                // Must match the layout planVulkanBarriers gives the private MSAA image.
                const VkImageLayout boundLayout = multisample && isDepthFormat(image->format)
                                                      ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                      : layout;
                graph.m_descriptorAllocator.writeImage(descriptorSet, use.binding, view, sampler, boundLayout, type);
            }

            for (const BufferUse &use : pass.bufferUses)
            {
                if (!use.isDescriptor())
                {
                    continue;
                }
                const std::string     &name   = graph.m_definition.name(use.buffer);
                const AllocatedBuffer *buffer = m_registry.getBuffer(name, slot);
                if (!buffer)
                {
                    throw std::runtime_error("FrameGraph: pass '" + pass.name + "' binds unknown buffer '" + name +
                                             "'");
                }
                const VkDescriptorType type = use.usage == BufferUsage::Uniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                                                : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                graph.m_descriptorAllocator.writeBuffer(descriptorSet, use.binding, buffer->buffer, 0, buffer->size,
                                                        type);
            }
        }
        graph.m_descriptorAllocator.commit();
    }
}

void FrameGraphCompiler::buildPipelines(CompiledFrameGraph &graph) const
{
    const auto passes = graph.m_definition.passes();
    for (size_t index = 0; index < passes.size(); ++index)
    {
        const PassDesc &pass = passes[index];
        if (pass.type == PassType::Custom)
        {
            continue;
        }
        if (pass.type == PassType::Compute)
        {
            if (pass.computeShader.empty())
            {
                throw std::runtime_error("FrameGraph: compute pass '" + pass.name + "' has no shader");
            }
            graph.m_passes[index].pipeline =
                std::make_unique<ComputePipeline>(m_ctx, pass.computeShader, graph.m_passes[index].pipelineLayout);
            continue;
        }
        if (pass.vertShader.empty() || pass.fragShader.empty())
        {
            throw std::runtime_error("FrameGraph: graphics pass '" + pass.name + "' has no shaders");
        }

        std::vector<VkFormat> colorFormats;
        VkFormat              depthFormat = VK_FORMAT_UNDEFINED;
        for (const ImageUse &use : pass.imageUses)
        {
            if (use.usage == ImageUsage::DepthAttachment)
            {
                depthFormat = use.format;
            } else if (use.usage == ImageUsage::ColorAttachment)
            {
                colorFormats.push_back(use.format);
            }
        }

        const GraphicsState &state = pass.graphics;
        if ((state.depthTest.value_or(false) || state.depthWrite.value_or(false)) && depthFormat == VK_FORMAT_UNDEFINED)
        {
            throw std::runtime_error("FrameGraph: pass '" + pass.name +
                                     "' enables depth test/write but has no depth attachment");
        }
        if (state.polygonMode != VK_POLYGON_MODE_FILL && !m_ctx.supportsWireframe())
        {
            throw std::runtime_error("FrameGraph: pass '" + pass.name +
                                     "' uses a line/point polygon mode, but this device lacks fillModeNonSolid");
        }

        GraphicsPipeline::Config config{};
        config.state                   = state;
        config.samples                 = state.samples;
        config.vertShader              = pass.vertShader;
        config.fragShader              = pass.fragShader;
        config.vertexBindings          = pass.vertexBindings;
        config.vertexAttributes        = pass.vertexAttributes;
        config.passType                = pass.type;
        config.topology                = pass.topology;
        config.colorAttachmentFormats  = std::move(colorFormats);
        config.depthAttachmentFormat   = depthFormat;
        config.layout                  = graph.m_passes[index].pipelineLayout;
        graph.m_passes[index].pipeline = std::make_unique<GraphicsPipeline>(m_ctx, config);
    }
}

void FrameGraphCompiler::buildBarriers(CompiledFrameGraph &graph) const
{
    std::unordered_map<std::string, VkImageLayout> initialLayouts;
    std::unordered_map<std::string, VkImageLayout> finalLayouts;
    const auto                                     passes = graph.m_definition.passes();
    for (const PassDesc &pass : passes)
    {
        for (const ImageUse &use : pass.imageUses)
        {
            const std::string &name = graph.m_definition.name(use.image);
            if (m_registry.getImage(name))
            {
                initialLayouts.try_emplace(name, m_registry.getImageLayout(name));
            } else if (m_registry.hasImageArray(name))
            {
                const auto layouts = m_registry.getImageArrayLayouts(name);
                if (!layouts.empty())
                {
                    const VkImageLayout common = layouts.front();
                    for (VkImageLayout layout : layouts)
                    {
                        if (layout != common)
                        {
                            throw std::runtime_error("FrameGraph: image array '" + name +
                                                     "' has mixed initial layouts");
                        }
                    }
                    initialLayouts.try_emplace(name, common);
                }
            }
        }
    }

    // An acquired backbuffer may be a never-presented swapchain image whose
    // layout is still UNDEFINED. Treat every frame as discard-on-entry: using
    // UNDEFINED is valid even after an image has previously been presented and
    // avoids carrying per-swapchain-image layout history into the graph. The
    // graph still returns every backbuffer to PRESENT_SRC_KHR before submission.
    for (const FrameGraphDefinition::ExternalImageDesc &external : graph.m_definition.externalImages())
    {
        const std::string &name = graph.m_definition.name(external.image);
        initialLayouts[name]    = VK_IMAGE_LAYOUT_UNDEFINED;
        finalLayouts[name]      = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }

    const auto plan = framegraph::planVulkanBarriers(passes, graph.m_definition.resources(), graph.m_sortedIndices,
                                                     initialLayouts, finalLayouts);
    for (size_t passIndex = 0; passIndex < plan.beforePass.size(); ++passIndex)
    {
        auto &compiled = graph.m_passes[passIndex];
        for (const framegraph::PlannedBarrier &planned : plan.beforePass[passIndex])
        {
            if (planned.kind != framegraph::BarrierResourceKind::Buffer)
            {
                const bool multisample = planned.kind == framegraph::BarrierResourceKind::MultisampleImage;
                std::vector<const AllocatedImage *> images;
                if (multisample)
                {
                    if (const AllocatedImage *image = graph.multisampleImage(planned.resourceName))
                    {
                        images.push_back(image);
                    }
                } else if (const AllocatedImage *image = m_registry.getImage(planned.resourceName))
                {
                    images.push_back(image);
                } else
                {
                    images = m_registry.getImageArray(planned.resourceName);
                }
                if (images.empty())
                {
                    throw std::runtime_error("FrameGraph: barrier references missing image '" + planned.resourceName +
                                             "'");
                }
                for (const AllocatedImage *image : images)
                {
                    if (!image)
                    {
                        throw std::runtime_error("FrameGraph: barrier references an empty image-array slot");
                    }
                    VkImageMemoryBarrier2 barrier{};
                    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    barrier.srcStageMask        = planned.source.stages;
                    barrier.srcAccessMask       = planned.source.access;
                    barrier.dstStageMask        = planned.destination.stages;
                    barrier.dstAccessMask       = planned.destination.access;
                    barrier.oldLayout           = planned.source.layout;
                    barrier.newLayout           = planned.destination.layout;
                    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.image               = image->image;
                    const VkImageAspectFlags aspect =
                        isDepthFormat(image->format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
                    barrier.subresourceRange = {aspect, 0, image->mipLevels, 0, image->arrayLayers};
                    // The prefix keeps a private image from matching an external image of the same name.
                    compiled.imageBarriers.push_back(
                        {barrier, multisample ? "__msaa_" + planned.resourceName : planned.resourceName});
                }
            } else
            {
                const AllocatedBuffer *buffer = m_registry.getBuffer(planned.resourceName);
                if (!buffer)
                {
                    throw std::runtime_error("FrameGraph: barrier references missing buffer '" + planned.resourceName +
                                             "'");
                }
                VkBufferMemoryBarrier2 barrier{};
                barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                barrier.srcStageMask        = planned.source.stages;
                barrier.srcAccessMask       = planned.source.access;
                barrier.dstStageMask        = planned.destination.stages;
                barrier.dstAccessMask       = planned.destination.access;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer              = buffer->buffer;
                barrier.offset              = 0;
                barrier.size                = VK_WHOLE_SIZE;
                compiled.bufferBarriers.push_back({barrier, planned.resourceName});
            }
        }
    }

    for (const framegraph::PlannedBarrier &planned : plan.afterGraph)
    {
        const AllocatedImage *image = m_registry.getImage(planned.resourceName);
        if (!image)
        {
            throw std::runtime_error("FrameGraph: final barrier references missing image '" + planned.resourceName +
                                     "'");
        }
        VkImageMemoryBarrier2 barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask        = planned.source.stages;
        barrier.srcAccessMask       = planned.source.access;
        barrier.dstStageMask        = planned.destination.stages;
        barrier.dstAccessMask       = planned.destination.access;
        barrier.oldLayout           = planned.source.layout;
        barrier.newLayout           = planned.destination.layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = image->image;
        const VkImageAspectFlags aspect =
            isDepthFormat(image->format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange = {aspect, 0, image->mipLevels, 0, image->arrayLayers};
        graph.m_finalImageBarriers.push_back({barrier, planned.resourceName});
    }

    // Applied to the registry by CompiledFrameGraph::execute(), once the graph actually records a frame.
    graph.m_finalImageLayouts = plan.finalImageLayouts;
}

} // namespace lr
