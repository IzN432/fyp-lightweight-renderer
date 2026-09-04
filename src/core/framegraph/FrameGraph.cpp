#include "FrameGraph.hpp"
#include "PassDescAdapter.hpp"
#include "FrameGraphTopology.hpp"
#include "core/vulkan/VkResultUtils.hpp"

#include "core/pipeline/ComputePipeline.hpp"
#include "core/pipeline/GraphicsPipeline.hpp"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <unordered_set>

namespace
{

bool isDepthFormat(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D16_UNORM ||
           format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

} // namespace

namespace lr
{

FrameGraph::FrameGraph(const VulkanContext &ctx, ResourceRegistry &registry)
    : m_ctx(ctx), m_registry(registry), m_descriptorAllocator(ctx.getDevice())
{
    createDefaultSampler();
}

FrameGraph::~FrameGraph()
{
    destroyCompiledPasses();
    vkDestroySampler(m_ctx.getDevice(), m_defaultSampler, nullptr);
}

// Add a pass with the given name and return a builder for configuring it. The pass won't be compiled until compile() is
// called.
PassBuilder FrameGraph::addPass(std::string name)
{
    const PassHandle handle = m_graph.addPass(std::move(name));
    return PassBuilder(m_graph, handle);
}

// Compiles the pass graph: topological sort, resource allocation, pipeline and descriptor set creation, barrier
// generation.
void FrameGraph::destroyCompiledPasses()
{
    // Pipelines must be explicitly destroyed — owned here via unique_ptr.
    // Descriptor set layouts and pipeline layouts are owned by m_descriptorAllocator;
    // do not destroy them here.
    for (auto &cp : m_compiled)
    {
        cp.pipeline.reset();
    }
}

void FrameGraph::compile()
{
    auto passes = m_graph.passes();
    spdlog::info("FrameGraph: compiling {} passes...", passes.size());

    destroyCompiledPasses();
    m_descriptorAllocator.reset();
    m_compiled.clear();
    m_compiled.resize(passes.size());

    sortPasses();
    allocateResources();
    buildDescriptorSets();
    buildPipelines();
    buildBarriers();

    if (spdlog::should_log(spdlog::level::debug))
    {
        spdlog::debug("{}", debugDump());
    }
    spdlog::info("FrameGraph: compiled OK");
}

std::array<float, 4> FrameGraph::debugLabelColor(PassType type)
{
    switch (type)
    {
        case PassType::Geometry:
            return {0.20f, 0.70f, 1.00f, 1.00f};
        case PassType::Fullscreen:
            return {0.20f, 1.00f, 0.50f, 1.00f};
        case PassType::Compute:
            return {1.00f, 0.65f, 0.20f, 1.00f};
        case PassType::Custom:
            return {0.85f, 0.40f, 1.00f, 1.00f};
    }
    return {};
}

void FrameGraph::submitResourceBarriers(CommandBuffer &cmd, const CompiledPass &compiled)
{
    if (compiled.imageBarriers.empty() && compiled.bufferBarriers.empty())
    {
        return;
    }

    // Patch external image handles and submit all resource barriers together.
    m_scratchImageBarriers.clear();
    for (const auto &cb : compiled.imageBarriers)
    {
        VkImageMemoryBarrier2 b  = cb.barrier;
        auto                  it = m_externalImages.find(cb.resourceName);
        if (it != m_externalImages.end())
        {
            b.image = it->second.image;
        }
        m_scratchImageBarriers.push_back(b);
    }

    m_scratchBufferBarriers.clear();
    for (const auto &cb : compiled.bufferBarriers)
    {
        m_scratchBufferBarriers.push_back(cb.barrier);
    }

    VkDependencyInfo dep{};
    dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount  = static_cast<uint32_t>(m_scratchImageBarriers.size());
    dep.pImageMemoryBarriers     = m_scratchImageBarriers.data();
    dep.bufferMemoryBarrierCount = static_cast<uint32_t>(m_scratchBufferBarriers.size());
    dep.pBufferMemoryBarriers    = m_scratchBufferBarriers.data();
    vkCmdPipelineBarrier2(cmd.get(), &dep);
}

VkRenderingInfo FrameGraph::prepareRenderingInfo(const PassDesc &pass, VkExtent2D extent)
{
    m_scratchColorAttachments.clear();
    m_scratchDepthAttachment = {};
    bool hasDepth            = false;

    for (const ImageUse &write : pass.imageUses)
    {
        if (!write.isAttachment())
        {
            continue;
        }
        const std::string    &name = m_graph.name(write.image);
        const AllocatedImage *img  = m_registry.getImage(name);

        // External images (e.g. swapchain) have their view injected per-frame
        VkImageView view  = img->view;
        auto        extIt = m_externalImages.find(name);
        if (extIt != m_externalImages.end())
        {
            view = extIt->second.view;
        }

        VkRenderingAttachmentInfo ai{};
        ai.sType      = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        ai.imageView  = view;
        ai.loadOp     = write.loadOp;
        ai.storeOp    = write.storeOp;
        ai.clearValue = write.clearValue;

        if (write.usage == ImageUsage::DepthAttachment)
        {
            ai.imageLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            m_scratchDepthAttachment = ai;
            hasDepth                 = true;
        } else
        {
            ai.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            m_scratchColorAttachments.push_back(ai);
        }
    }

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea           = {{0, 0}, extent};
    renderingInfo.layerCount           = 1;
    renderingInfo.colorAttachmentCount = static_cast<uint32_t>(m_scratchColorAttachments.size());
    renderingInfo.pColorAttachments    = m_scratchColorAttachments.data();
    renderingInfo.pDepthAttachment     = hasDepth ? &m_scratchDepthAttachment : nullptr;
    return renderingInfo;
}

void FrameGraph::bindVertexAndIndexBuffers(CommandBuffer &cmd, const PassDesc &pass)
{
    for (const BufferUse &use : pass.bufferUses)
    {
        if (use.usage != BufferUsage::Vertex)
        {
            continue;
        }
        const std::string     &name = m_graph.name(use.buffer);
        const AllocatedBuffer *vb   = m_registry.getBuffer(name);
        if (!vb)
        {
            spdlog::error("FrameGraph::execute: pass '{}' references missing vertex buffer '{}'", pass.name, name);
            throw std::runtime_error("FrameGraph::execute: pass '" + pass.name + "' references vertex buffer '" + name +
                                     "' which was not registered");
        }
        cmd.bindVertexBuffer(use.binding, vb->buffer);
    }

    for (const BufferUse &use : pass.bufferUses)
    {
        if (use.usage != BufferUsage::Index)
        {
            continue;
        }
        const std::string     &name = m_graph.name(use.buffer);
        const AllocatedBuffer *ib   = m_registry.getBuffer(name);
        if (!ib)
        {
            spdlog::error("FrameGraph::execute: pass '{}' references missing index buffer '{}'", pass.name, name);
            throw std::runtime_error("FrameGraph::execute: pass '" + pass.name + "' references index buffer '" + name +
                                     "' which was not registered");
        }
        cmd.bindIndexBuffer(ib->buffer);
    }
}

void FrameGraph::execute(CommandBuffer &cmd)
{
    m_registry.flushUploads();

    for (size_t idx : m_sortedIndices)
    {
        const PassDesc     &pass     = m_graph.passes()[idx];
        const CompiledPass &compiled = m_compiled[idx];

        const bool useDebugLabels = m_ctx.debugNamesEnabled();
        if (useDebugLabels)
        {
            m_ctx.beginDebugLabel(cmd.get(), pass.name, debugLabelColor(pass.type));
        }

        submitResourceBarriers(cmd, compiled);

        // Bind pipeline and descriptor set — common to both compute and graphics
        if (compiled.pipeline)
        {
            vkCmdBindPipeline(cmd.get(), compiled.pipeline->bindPoint(), compiled.pipeline->get());
        }

        if (compiled.descriptorSet != VK_NULL_HANDLE && compiled.pipeline)
        {
            vkCmdBindDescriptorSets(cmd.get(), compiled.pipeline->bindPoint(), compiled.pipelineLayout, 0, 1,
                                    &compiled.descriptorSet, 0, nullptr);
        }

        PassContext context(cmd, compiled.pipelineLayout, compiled.renderingExtent, m_graph, m_registry);

        // Compute and Custom passes skip dynamic rendering — just invoke callback and move on
        if (pass.type == PassType::Compute || pass.type == PassType::Custom)
        {
            if (pass.executeCallback)
            {
                pass.executeCallback(context);
            }

            if (useDebugLabels)
            {
                m_ctx.endDebugLabel(cmd.get());
            }
            continue;
        }

        // Graphics passes: begin rendering, set dynamic state, draw, end rendering
        VkRenderingInfo renderingInfo = prepareRenderingInfo(pass, compiled.renderingExtent);
        vkCmdBeginRendering(cmd.get(), &renderingInfo);

        cmd.setViewport(0.0f, 0.0f, static_cast<float>(compiled.renderingExtent.width),
                        static_cast<float>(compiled.renderingExtent.height));
        cmd.setScissor(0, 0, compiled.renderingExtent.width, compiled.renderingExtent.height);

        bindVertexAndIndexBuffers(cmd, pass);

        if (pass.executeCallback)
        {
            pass.executeCallback(context);
        }

        vkCmdEndRendering(cmd.get());

        if (useDebugLabels)
        {
            m_ctx.endDebugLabel(cmd.get());
        }
    }
}

void FrameGraph::resize(VkExtent2D newExtent)
{
    m_registry.rebuild(newExtent);
    compile();
}

// ---------------------------------------------------------------------------
// Compilation steps - stubs to be implemented
// ---------------------------------------------------------------------------

void FrameGraph::sortPasses()
{
    const auto passes = m_graph.passes();
    m_logicalGraph    = framegraph::translatePassDescriptions(passes, m_graph.resources(), m_graph.owner());
    m_executionPlan   = framegraph::buildExecutionPlan(m_logicalGraph);

    m_sortedIndices.clear();
    m_sortedIndices.reserve(m_executionPlan.orderedPasses.size());
    for (framegraph::PassId pass : m_executionPlan.orderedPasses)
    {
        m_sortedIndices.push_back(pass.value);
    }

    spdlog::debug("FrameGraph: pass order:");
    for (size_t idx : m_sortedIndices)
    {
        spdlog::debug("  [{}] {}", idx, passes[idx].name);
    }
}

void FrameGraph::allocateResources()
{
    const auto planned = framegraph::planAttachmentImages(m_graph.passes(), m_graph.resources());
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

    const auto renderingExtents = framegraph::planRenderingExtents(m_graph.passes(), m_registry.getExtent());
    for (size_t passIndex = 0; passIndex < renderingExtents.size(); ++passIndex)
    {
        m_compiled[passIndex].renderingExtent = renderingExtents[passIndex];
    }

    for (const PassDesc &pass : m_graph.passes())
    {
        for (const ImageUse &use : pass.imageUses)
        {
            if (use.isAttachment())
            {
                continue;
            }
            const VkImageUsageFlags requiredUsage =
                use.usage == ImageUsage::Storage ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT;
            m_registry.validateImageUsage(m_graph.name(use.image), requiredUsage);
        }
    }
}

void FrameGraph::buildDescriptorSets()
{
    const auto passes = m_graph.passes();
    for (size_t i = 0; i < passes.size(); ++i)
    {
        const PassDesc &pass     = passes[i];
        CompiledPass   &compiled = m_compiled[i];

        if (pass.type == PassType::Custom)
        {
            spdlog::debug("FrameGraph: pass '{}' - custom (no descriptors or pipeline)", pass.name);
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
            const VkDescriptorType type = use.usage == BufferUsage::Uniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                                            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            addLayoutBinding(use.binding, type, 1, use.stages);
        }

        compiled.descriptorLayout = m_descriptorAllocator.createLayout(layoutBindings);
        compiled.pipelineLayout   = m_descriptorAllocator.createPipelineLayout(
            compiled.descriptorLayout, pass.pushConstantSize, pass.pushConstantStages);

        if (layoutBindings.empty())
        {
            spdlog::debug("FrameGraph: pass '{}' - no bindings", pass.name);
            continue;
        }

        compiled.descriptorSet = m_descriptorAllocator.allocate(compiled.descriptorLayout);

        for (const ImageUse &use : pass.imageUses)
        {
            if (!use.isDescriptor())
            {
                continue;
            }
            const std::string     &name    = m_graph.name(use.image);
            const bool             storage = use.usage == ImageUsage::Storage;
            const VkDescriptorType type =
                storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            const VkSampler     sampler = storage ? VK_NULL_HANDLE : m_defaultSampler;
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
                for (uint32_t index = 0; index < use.descriptorCount; ++index)
                {
                    if (!images[index])
                    {
                        throw std::runtime_error("FrameGraph: image array '" + name + "' has an empty slot");
                    }
                    views.push_back(images[index]->view);
                }
                m_descriptorAllocator.writeImageArray(compiled.descriptorSet, use.binding, views, sampler, layout,
                                                      type);
                continue;
            }

            const AllocatedImage *image = m_registry.getImage(name);
            if (!image)
            {
                throw std::runtime_error("FrameGraph: pass '" + pass.name + "' binds unknown image '" + name + "'");
            }
            VkImageView view = image->view;
            if (use.boundMip != allImageMips)
            {
                if (!storage)
                {
                    throw std::runtime_error("FrameGraph: mip-specific views are only supported for storage images");
                }
                if (use.boundMip >= image->mipViews.size())
                {
                    throw std::runtime_error("FrameGraph: pass '" + pass.name + "' requests invalid mip for '" + name +
                                             "'");
                }
                view = image->mipViews[use.boundMip];
            }
            m_descriptorAllocator.writeImage(compiled.descriptorSet, use.binding, view, sampler, layout, type);
        }

        for (const BufferUse &use : pass.bufferUses)
        {
            if (!use.isDescriptor())
            {
                continue;
            }
            const std::string     &name   = m_graph.name(use.buffer);
            const AllocatedBuffer *buffer = m_registry.getBuffer(name);
            if (!buffer)
            {
                throw std::runtime_error("FrameGraph: pass '" + pass.name + "' binds unknown buffer '" + name + "'");
            }
            const VkDescriptorType type = use.usage == BufferUsage::Uniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                                            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            m_descriptorAllocator.writeBuffer(compiled.descriptorSet, use.binding, buffer->buffer, 0, buffer->size,
                                              type);
        }

        m_descriptorAllocator.commit();
        spdlog::debug("FrameGraph: pass '{}' - {} binding(s)", pass.name, layoutBindings.size());
    }
}

void FrameGraph::buildPipelines()
{
    const auto passes = m_graph.passes();
    for (size_t i = 0; i < passes.size(); ++i)
    {
        const PassDesc &pass = passes[i];

        if (pass.type == PassType::Custom)
        {
            continue;
        }

        if (pass.type == PassType::Compute)
        {
            if (pass.computeShader.empty())
            {
                spdlog::error("FrameGraph: compute pass '{}' has no shader", pass.name);
                throw std::runtime_error("FrameGraph: compute pass '" + pass.name + "' has no shader");
            }

            m_compiled[i].pipeline =
                std::make_unique<ComputePipeline>(m_ctx, pass.computeShader, m_compiled[i].pipelineLayout);
            spdlog::debug("FrameGraph: pass '{}' - compute pipeline built", pass.name);
            continue;
        }

        if (pass.vertShader.empty() || pass.fragShader.empty())
        {
            spdlog::error("FrameGraph: graphics pass '{}' has missing shaders (vert='{}', frag='{}')", pass.name,
                          pass.vertShader, pass.fragShader);
            throw std::runtime_error("FrameGraph: graphics pass '" + pass.name + "' has no shaders");
        }

        // Derive attachment formats from declared writes
        std::vector<VkFormat> colorFormats;
        VkFormat              depthFormat = VK_FORMAT_UNDEFINED;

        for (const ImageUse &write : pass.imageUses)
        {
            if (!write.isAttachment())
            {
                continue;
            }
            if (write.usage == ImageUsage::DepthAttachment)
            {
                depthFormat = write.format;
            } else
            {
                colorFormats.push_back(write.format);
            }
        }

        GraphicsPipeline::Config cfg{};
        cfg.vertShaderPath         = pass.vertShader;
        cfg.fragShaderPath         = pass.fragShader;
        cfg.vertexBindings         = pass.vertexBindings;
        cfg.vertexAttributes       = pass.vertexAttributes;
        cfg.passType               = pass.type;
        cfg.topology               = pass.topology;
        cfg.colorAttachmentFormats = std::move(colorFormats);
        cfg.depthAttachmentFormat  = depthFormat;
        cfg.layout                 = m_compiled[i].pipelineLayout;

        m_compiled[i].pipeline = std::make_unique<GraphicsPipeline>(m_ctx, cfg);
        spdlog::debug("FrameGraph: pass '{}' - pipeline built", pass.name);
    }
}

void FrameGraph::buildBarriers()
{
    std::unordered_map<std::string, VkImageLayout> initialImageLayouts;
    const auto                                     passes = m_graph.passes();
    for (const PassDesc &pass : passes)
    {
        for (const ImageUse &use : pass.imageUses)
        {
            const std::string &name = m_graph.name(use.image);
            if (m_registry.getImage(name))
            {
                initialImageLayouts.try_emplace(name, m_registry.getImageLayout(name));
            } else if (m_registry.hasImageArray(name))
            {
                const std::vector<VkImageLayout> layouts = m_registry.getImageArrayLayouts(name);
                if (!layouts.empty())
                {
                    const VkImageLayout commonLayout = layouts.front();
                    for (VkImageLayout layout : layouts)
                    {
                        if (layout != commonLayout)
                        {
                            throw std::runtime_error("FrameGraph: image array '" + name +
                                                     "' has mixed initial layouts");
                        }
                    }
                    initialImageLayouts.try_emplace(name, commonLayout);
                }
            }
        }
    }

    const framegraph::VulkanBarrierPlan plan =
        framegraph::planVulkanBarriers(passes, m_graph.resources(), m_sortedIndices, initialImageLayouts);

    for (size_t passIndex = 0; passIndex < plan.beforePass.size(); ++passIndex)
    {
        CompiledPass &compiled = m_compiled[passIndex];
        for (const framegraph::PlannedBarrier &planned : plan.beforePass[passIndex])
        {
            if (planned.kind == framegraph::BarrierResourceKind::Image)
            {
                std::vector<const AllocatedImage *> images;
                if (const AllocatedImage *image = m_registry.getImage(planned.resourceName))
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
                        throw std::runtime_error("FrameGraph: barrier references empty slot in image array '" +
                                                 planned.resourceName + "'");
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
                    compiled.imageBarriers.push_back({barrier, planned.resourceName});
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

    for (const auto &[name, layout] : plan.finalImageLayouts)
    {
        if (m_registry.hasImageArray(name))
        {
            m_registry.setImageArrayLayout(name, layout);
        } else
        {
            m_registry.setImageLayout(name, layout);
        }
    }
}

void FrameGraph::setExternalImage(const std::string &name, VkImage image, VkImageView view)
{
    m_externalImages[name] = {image, view};
}

void FrameGraph::executeAndWait(std::vector<FinalLayoutDesc> finalLayouts)
{
    compile();

    VkDevice device = m_ctx.getDevice();

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.queueFamilyIndex = static_cast<uint32_t>(m_ctx.getGraphicsQueueFamily());
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

    VkCommandPool pool = VK_NULL_HANDLE;
    checkVk(vkCreateCommandPool(device, &poolCI, nullptr, &pool), "FrameGraph::executeAndWait: vkCreateCommandPool");

    VkFence                      fence     = VK_NULL_HANDLE;
    bool                         submitted = false;
    std::vector<FinalLayoutDesc> completedLayouts;
    try
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool        = pool;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer vkCmd = VK_NULL_HANDLE;
        checkVk(vkAllocateCommandBuffers(device, &allocInfo, &vkCmd),
                "FrameGraph::executeAndWait: vkAllocateCommandBuffers");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        checkVk(vkBeginCommandBuffer(vkCmd, &beginInfo), "FrameGraph::executeAndWait: vkBeginCommandBuffer");

        CommandBuffer cmd(vkCmd);
        execute(cmd);

        if (!finalLayouts.empty())
        {
            std::vector<VkImageMemoryBarrier2> barriers;
            barriers.reserve(finalLayouts.size());

            for (const auto &fl : finalLayouts)
            {
                const AllocatedImage *img = m_registry.getImage(fl.resourceName);
                if (!img)
                {
                    continue;
                }

                VkImageLayout current = m_registry.getImageLayout(fl.resourceName);
                if (current == fl.layout)
                {
                    continue;
                }

                VkImageAspectFlags aspect =
                    isDepthFormat(img->format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

                VkImageMemoryBarrier2 barrier{};
                barrier.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                barrier.srcStageMask     = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                barrier.srcAccessMask    = VK_ACCESS_2_SHADER_WRITE_BIT;
                barrier.dstStageMask     = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                barrier.dstAccessMask    = VK_ACCESS_2_SHADER_READ_BIT;
                barrier.oldLayout        = current;
                barrier.newLayout        = fl.layout;
                barrier.image            = img->image;
                barrier.subresourceRange = {aspect, 0, img->mipLevels, 0, img->arrayLayers};
                barriers.push_back(barrier);

                completedLayouts.push_back(fl);
            }

            if (!barriers.empty())
            {
                VkDependencyInfo dep{};
                dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
                dep.pImageMemoryBarriers    = barriers.data();
                vkCmdPipelineBarrier2(vkCmd, &dep);
            }
        }

        checkVk(vkEndCommandBuffer(vkCmd), "FrameGraph::executeAndWait: vkEndCommandBuffer");

        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        checkVk(vkCreateFence(device, &fenceCI, nullptr, &fence), "FrameGraph::executeAndWait: vkCreateFence");

        VkCommandBufferSubmitInfo cmdSubmit{};
        cmdSubmit.sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cmdSubmit.commandBuffer = vkCmd;

        VkSubmitInfo2 submitInfo{};
        submitInfo.sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos    = &cmdSubmit;

        checkVk(vkQueueSubmit2(m_ctx.getGraphicsQueue(), 1, &submitInfo, fence),
                "FrameGraph::executeAndWait: vkQueueSubmit2");
        submitted = true;
        checkVk(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "FrameGraph::executeAndWait: vkWaitForFences");
    } catch (...)
    {
        if (submitted)
        {
            (void)vkDeviceWaitIdle(device);
        }
        if (fence != VK_NULL_HANDLE)
        {
            vkDestroyFence(device, fence, nullptr);
        }
        vkDestroyCommandPool(device, pool, nullptr);
        throw;
    }

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, pool, nullptr); // also frees vkCmd

    for (const auto &layout : completedLayouts)
    {
        m_registry.setImageLayout(layout.resourceName, layout.layout);
    }

    spdlog::info("FrameGraph: synchronous execution complete");
}

std::vector<PassHandle> FrameGraph::passHandles() const { return m_graph.passHandles(); }

std::string FrameGraph::debugDump() const
{
    const auto                                             passes = m_graph.passes();
    std::vector<std::vector<framegraph::BarrierDebugInfo>> barriersByPass(passes.size());
    for (size_t passIndex = 0; passIndex < m_compiled.size(); ++passIndex)
    {
        auto &output = barriersByPass[passIndex];
        output.reserve(m_compiled[passIndex].imageBarriers.size() + m_compiled[passIndex].bufferBarriers.size());
        for (const auto &compiledBarrier : m_compiled[passIndex].imageBarriers)
        {
            const auto &barrier = compiledBarrier.barrier;
            output.push_back({
                .resourceName = compiledBarrier.resourceName,
                .srcStage     = barrier.srcStageMask,
                .srcAccess    = barrier.srcAccessMask,
                .dstStage     = barrier.dstStageMask,
                .dstAccess    = barrier.dstAccessMask,
                .oldLayout    = barrier.oldLayout,
                .newLayout    = barrier.newLayout,
            });
        }
        for (const auto &compiledBarrier : m_compiled[passIndex].bufferBarriers)
        {
            const auto &barrier = compiledBarrier.barrier;
            output.push_back({
                .resourceName = compiledBarrier.resourceName,
                .srcStage     = barrier.srcStageMask,
                .srcAccess    = barrier.srcAccessMask,
                .dstStage     = barrier.dstStageMask,
                .dstAccess    = barrier.dstAccessMask,
            });
        }
    }

    return framegraph::dumpTopology(passes, m_graph.resources(), m_sortedIndices, barriersByPass);
}

void FrameGraph::createDefaultSampler()
{
    VkSamplerCreateInfo ci{};
    ci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.magFilter    = VK_FILTER_LINEAR;
    ci.minFilter    = VK_FILTER_LINEAR;
    ci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.minLod       = 0.0f;
    ci.maxLod       = VK_LOD_CLAMP_NONE;

    checkVk(vkCreateSampler(m_ctx.getDevice(), &ci, nullptr, &m_defaultSampler), "FrameGraph: vkCreateSampler");
}

} // namespace lr
