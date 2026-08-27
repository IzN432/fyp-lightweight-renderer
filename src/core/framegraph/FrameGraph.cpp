#include "FrameGraph.hpp"
#include "PassDescAdapter.hpp"
#include "FrameGraphTopology.hpp"
#include "core/vulkan/VkResultUtils.hpp"

#include "core/pipeline/ComputePipeline.hpp"
#include "core/pipeline/GraphicsPipeline.hpp"

#include <spdlog/spdlog.h>

#include <stdexcept>

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
PassBuilder FrameGraph::addPass(const std::string &name)
{
    return PassBuilder(framegraph::appendPass(m_passes, name));
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
    spdlog::info("FrameGraph: compiling {} passes...", m_passes.size());

    destroyCompiledPasses();
    m_descriptorAllocator.reset();
    m_compiled.clear();
    m_compiled.resize(m_passes.size());

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

void FrameGraph::execute(CommandBuffer &cmd)
{
    m_registry.flushUploads();

    VkExtent2D extent = m_registry.getExtent();

    for (size_t idx : m_sortedIndices)
    {
        const PassDesc     &pass     = m_passes[idx];
        const CompiledPass &compiled = m_compiled[idx];

        const bool useDebugLabels = m_ctx.debugNamesEnabled();
        if (useDebugLabels)
        {
            std::array<float, 4> color{};

            switch (pass.type)
            {
                case PassType::Geometry:
                    color = {0.20f, 0.70f, 1.00f, 1.00f};
                    break;
                case PassType::Fullscreen:
                    color = {0.20f, 1.00f, 0.50f, 1.00f};
                    break;
                case PassType::Compute:
                    color = {1.00f, 0.65f, 0.20f, 1.00f};
                    break;
                case PassType::Custom:
                    color = {0.85f, 0.40f, 1.00f, 1.00f};
                    break;
            }

            m_ctx.beginDebugLabel(cmd.get(), pass.name, color);
        }

        // Patch external image handles and submit all resource barriers together.
        if (!compiled.imageBarriers.empty() || !compiled.bufferBarriers.empty())
        {
            std::vector<VkImageMemoryBarrier2> patchedBarriers;
            patchedBarriers.reserve(compiled.imageBarriers.size());
            for (const auto &cb : compiled.imageBarriers)
            {
                VkImageMemoryBarrier2 b  = cb.barrier;
                auto                  it = m_externalImages.find(cb.resourceName);
                if (it != m_externalImages.end())
                {
                    b.image = it->second.image;
                }
                patchedBarriers.push_back(b);
            }

            std::vector<VkBufferMemoryBarrier2> bufferBarriers;
            bufferBarriers.reserve(compiled.bufferBarriers.size());
            for (const auto &cb : compiled.bufferBarriers)
            {
                bufferBarriers.push_back(cb.barrier);
            }

            VkDependencyInfo dep{};
            dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = static_cast<uint32_t>(patchedBarriers.size());
            dep.pImageMemoryBarriers    = patchedBarriers.data();
            dep.bufferMemoryBarrierCount = static_cast<uint32_t>(bufferBarriers.size());
            dep.pBufferMemoryBarriers    = bufferBarriers.data();
            vkCmdPipelineBarrier2(cmd.get(), &dep);
        }

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

        // Compute and Custom passes skip dynamic rendering — just invoke callback and move on
        if (pass.type == PassType::Compute || pass.type == PassType::Custom)
        {
            if (pass.executeCallback)
            {
                pass.executeCallback(cmd, compiled.pipelineLayout);
            }

            if (useDebugLabels)
            {
                m_ctx.endDebugLabel(cmd.get());
            }
            continue;
        }

        // Graphics passes: begin rendering, set dynamic state, draw, end rendering
        std::vector<VkRenderingAttachmentInfo> colorAttachments;
        VkRenderingAttachmentInfo              depthAttachment{};
        bool                                   hasDepth = false;

        for (const auto &write : pass.writes)
        {
            const AllocatedImage *img = m_registry.getImage(write.name);

            // External images (e.g. swapchain) have their view injected per-frame
            VkImageView view  = img->view;
            auto        extIt = m_externalImages.find(write.name);
            if (extIt != m_externalImages.end())
            {
                view = extIt->second.view;
            }

            VkRenderingAttachmentInfo ai{};
            ai.sType      = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            ai.imageView  = view;
            ai.loadOp     = write.loadOp;
            ai.storeOp    = VK_ATTACHMENT_STORE_OP_STORE;
            ai.clearValue = write.clearValue;

            if (isDepthFormat(write.format))
            {
                ai.imageLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                depthAttachment = ai;
                hasDepth        = true;
            } else
            {
                ai.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                colorAttachments.push_back(ai);
            }
        }

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea           = {{0, 0}, extent};
        renderingInfo.layerCount           = 1;
        renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size());
        renderingInfo.pColorAttachments    = colorAttachments.data();
        renderingInfo.pDepthAttachment     = hasDepth ? &depthAttachment : nullptr;

        vkCmdBeginRendering(cmd.get(), &renderingInfo);

        cmd.setViewport(0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height));
        cmd.setScissor(0, 0, extent.width, extent.height);

        // Bind vertex buffers declared by this pass
        for (const auto &vbRef : pass.vertexBufferRefs)
        {
            const AllocatedBuffer *vb = m_registry.getBuffer(vbRef.bufferName);
            if (!vb)
            {
                spdlog::error("FrameGraph::execute: pass '{}' references missing vertex buffer '{}'", pass.name,
                              vbRef.bufferName);
                throw std::runtime_error("FrameGraph::execute: pass '" + pass.name + "' references vertex buffer '" +
                                         vbRef.bufferName + "' which was not registered");
            }
            cmd.bindVertexBuffer(vbRef.binding, vb->buffer);
        }

        // Bind index buffer if declared
        if (!pass.indexBufferName.empty())
        {
            const AllocatedBuffer *ib = m_registry.getBuffer(pass.indexBufferName);
            if (!ib)
            {
                spdlog::error("FrameGraph::execute: pass '{}' references missing index buffer '{}'", pass.name,
                              pass.indexBufferName);
                throw std::runtime_error("FrameGraph::execute: pass '" + pass.name + "' references index buffer '" +
                                         pass.indexBufferName + "' which was not registered");
            }
            cmd.bindIndexBuffer(ib->buffer);
        }

        if (pass.executeCallback)
        {
            pass.executeCallback(cmd, compiled.pipelineLayout);
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
    m_definition = framegraph::translatePassDescriptions(m_passes);
    m_executionPlan = framegraph::buildExecutionPlan(m_definition);

    m_sortedIndices.clear();
    m_sortedIndices.reserve(m_executionPlan.orderedPasses.size());
    for (framegraph::PassId pass : m_executionPlan.orderedPasses)
    {
        m_sortedIndices.push_back(pass.value);
    }

    spdlog::debug("FrameGraph: pass order:");
    for (size_t idx : m_sortedIndices)
    {
        spdlog::debug("  [{}] {}", idx, m_passes[idx].name);
    }
}

void FrameGraph::allocateResources()
{
    const auto planned = framegraph::planAttachmentImages(m_passes, m_registry.getExtent());
    for (const auto &image : planned)
    {
        if (!m_registry.hasImage(image.name))
        {
            m_registry.registerImage(image.name, image.format, image.usage, image.extent, image.aspect);
        }
    }
}

void FrameGraph::buildDescriptorSets()
{
    for (size_t i = 0; i < m_passes.size(); ++i)
    {
        const PassDesc &pass     = m_passes[i];
        CompiledPass   &compiled = m_compiled[i];

        if (pass.type == PassType::Custom)
        {
            spdlog::debug("FrameGraph: pass '{}' - custom (no descriptors or pipeline)", pass.name);
            continue;
        }

        // Build VkDescriptorSetLayoutBinding array
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        layoutBindings.reserve(pass.bindings.size());
        for (const auto &b : pass.bindings)
        {
            VkDescriptorSetLayoutBinding lb{};
            lb.binding         = b.binding;
            lb.descriptorType  = b.type;
            lb.descriptorCount = b.descriptorCount;
            lb.stageFlags      = b.stages;
            layoutBindings.push_back(lb);
        }

        compiled.descriptorLayout = m_descriptorAllocator.createLayout(layoutBindings);
        compiled.pipelineLayout   = m_descriptorAllocator.createPipelineLayout(
            compiled.descriptorLayout, pass.pushConstantSize, pass.pushConstantStages);

        if (pass.bindings.empty())
        {
            spdlog::debug("FrameGraph: pass '{}' - no bindings", pass.name);
            continue;
        }

        compiled.descriptorSet = m_descriptorAllocator.allocate(compiled.descriptorLayout);

        for (const auto &b : pass.bindings)
        {
            bool isImage = (b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
                            b.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || b.type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);

            if (isImage)
            {
                VkSampler sampler = (b.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) ? VK_NULL_HANDLE : m_defaultSampler;

                // Storage images must always be in GENERAL layout
                VkImageLayout layout =
                    (b.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) ? VK_IMAGE_LAYOUT_GENERAL : b.imageLayout;

                // Try to get as array first (arrays can have any size, including 1)
                const auto images = m_registry.getImageArray(b.resourceName);
                // For storage images, Vulkan requires levelCount == 1 in the image
                // view. Use a per-mip view when available (created by registerCubemap
                // or createMipViews). Fall back to the full-range view for images
                // with only one mip level.
                auto pickView = [&](const AllocatedImage *img) -> VkImageView {
                    if (b.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && !img->mipViews.empty())
                    {
                        if (b.mipLevel >= img->mipViews.size())
                        {
                            throw std::runtime_error("FrameGraph: pass '" + pass.name +
                                                     "' requests mipLevel=" + std::to_string(b.mipLevel) + " for '" +
                                                     b.resourceName + "' but image only has " +
                                                     std::to_string(img->mipViews.size()) + " mip view(s)");
                        }
                        return img->mipViews[b.mipLevel];
                    }
                    return img->view;
                };

                if (!images.empty())
                {
                    // It's an image array
                    if (images.size() < b.descriptorCount)
                    {
                        spdlog::error("FrameGraph: pass '{}' binds image array '{}' with descriptorCount={} "
                                      "but only {} slot(s) exist",
                                      pass.name, b.resourceName, b.descriptorCount, images.size());
                        throw std::runtime_error("FrameGraph: pass '" + pass.name + "' binds image array '" +
                                                 b.resourceName +
                                                 "' with descriptorCount=" + std::to_string(b.descriptorCount) +
                                                 " but only " + std::to_string(images.size()) + " slot(s) exist");
                    }

                    std::vector<VkImageView> views;
                    views.reserve(b.descriptorCount);
                    for (uint32_t idx = 0; idx < b.descriptorCount; ++idx)
                    {
                        const AllocatedImage *img = images[idx];
                        if (!img)
                        {
                            spdlog::error("FrameGraph: pass '{}' binds image array '{}' with empty slot at index {}",
                                          pass.name, b.resourceName, idx);
                            throw std::runtime_error("FrameGraph: pass '" + pass.name + "' binds image array '" +
                                                     b.resourceName + "' with an empty slot at index " +
                                                     std::to_string(idx));
                        }
                        views.push_back(pickView(img));
                    }

                    m_descriptorAllocator.writeImageArray(compiled.descriptorSet, b.binding, views, sampler, layout,
                                                          b.type);
                } else
                {
                    // Single image
                    const AllocatedImage *img = m_registry.getImage(b.resourceName);
                    if (!img)
                    {
                        spdlog::error("FrameGraph: pass '{}' binds unknown image '{}'", pass.name, b.resourceName);
                        throw std::runtime_error("FrameGraph: pass '" + pass.name + "' binds unknown image '" +
                                                 b.resourceName + "'");
                    }

                    m_descriptorAllocator.writeImage(compiled.descriptorSet, b.binding, pickView(img), sampler, layout,
                                                     b.type);
                }
            } else
            {
                if (b.descriptorCount != 1)
                {
                    spdlog::error(
                        "FrameGraph: pass '{}' uses unsupported descriptorCount={} for non-image binding '{}'",
                        pass.name, b.descriptorCount, b.resourceName);
                    throw std::runtime_error("FrameGraph: pass '" + pass.name + "' uses descriptorCount=" +
                                             std::to_string(b.descriptorCount) + " for non-image binding '" +
                                             b.resourceName + "', which is not supported yet");
                }

                const AllocatedBuffer *buf = m_registry.getBuffer(b.resourceName);
                if (!buf)
                {
                    spdlog::error("FrameGraph: pass '{}' binds unknown buffer '{}'", pass.name, b.resourceName);
                    throw std::runtime_error("FrameGraph: pass '" + pass.name + "' binds unknown buffer '" +
                                             b.resourceName + "'");
                }

                m_descriptorAllocator.writeBuffer(compiled.descriptorSet, b.binding, buf->buffer, 0, buf->size, b.type);
            }
        }

        m_descriptorAllocator.commit();
        spdlog::debug("FrameGraph: pass '{}' - {} binding(s)", pass.name, pass.bindings.size());
    }
}

void FrameGraph::buildPipelines()
{
    for (size_t i = 0; i < m_passes.size(); ++i)
    {
        const PassDesc &pass = m_passes[i];

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

        for (const auto &write : pass.writes)
        {
            if (isDepthFormat(write.format))
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
    for (const PassDesc &pass : m_passes)
    {
        for (const BindingDesc &binding : pass.bindings)
        {
            if (m_registry.getImage(binding.resourceName))
            {
                initialImageLayouts.try_emplace(
                    binding.resourceName, m_registry.getImageLayout(binding.resourceName));
            }
            else if (m_registry.hasImageArray(binding.resourceName))
            {
                const std::vector<VkImageLayout> layouts =
                    m_registry.getImageArrayLayouts(binding.resourceName);
                if (!layouts.empty())
                {
                    const VkImageLayout commonLayout = layouts.front();
                    for (VkImageLayout layout : layouts)
                    {
                        if (layout != commonLayout)
                        {
                            throw std::runtime_error(
                                "FrameGraph: image array '" + binding.resourceName +
                                "' has mixed initial layouts");
                        }
                    }
                    initialImageLayouts.try_emplace(binding.resourceName, commonLayout);
                }
            }
        }
        for (const ResourceDesc &attachment : pass.writes)
        {
            if (m_registry.getImage(attachment.name))
            {
                initialImageLayouts.try_emplace(
                    attachment.name, m_registry.getImageLayout(attachment.name));
            }
        }
    }

    const framegraph::VulkanBarrierPlan plan =
        framegraph::planVulkanBarriers(m_passes, m_sortedIndices, initialImageLayouts);

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
                }
                else
                {
                    images = m_registry.getImageArray(planned.resourceName);
                }
                if (images.empty())
                {
                    throw std::runtime_error("FrameGraph: barrier references missing image '" +
                                             planned.resourceName + "'");
                }

                for (const AllocatedImage *image : images)
                {
                    if (!image)
                    {
                        throw std::runtime_error(
                            "FrameGraph: barrier references empty slot in image array '" +
                            planned.resourceName + "'");
                    }

                    VkImageMemoryBarrier2 barrier{};
                    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    barrier.srcStageMask = planned.source.stages;
                    barrier.srcAccessMask = planned.source.access;
                    barrier.dstStageMask = planned.destination.stages;
                    barrier.dstAccessMask = planned.destination.access;
                    barrier.oldLayout = planned.source.layout;
                    barrier.newLayout = planned.destination.layout;
                    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.image = image->image;
                    const VkImageAspectFlags aspect = isDepthFormat(image->format)
                        ? VK_IMAGE_ASPECT_DEPTH_BIT
                        : VK_IMAGE_ASPECT_COLOR_BIT;
                    barrier.subresourceRange = {aspect, 0, image->mipLevels, 0, image->arrayLayers};
                    compiled.imageBarriers.push_back({barrier, planned.resourceName});
                }
            }
            else
            {
                const AllocatedBuffer *buffer = m_registry.getBuffer(planned.resourceName);
                if (!buffer)
                {
                    throw std::runtime_error("FrameGraph: barrier references missing buffer '" +
                                             planned.resourceName + "'");
                }

                VkBufferMemoryBarrier2 barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                barrier.srcStageMask = planned.source.stages;
                barrier.srcAccessMask = planned.source.access;
                barrier.dstStageMask = planned.destination.stages;
                barrier.dstAccessMask = planned.destination.access;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = buffer->buffer;
                barrier.offset = 0;
                barrier.size = VK_WHOLE_SIZE;
                compiled.bufferBarriers.push_back({barrier, planned.resourceName});
            }
        }
    }

    for (const auto &[name, layout] : plan.finalImageLayouts)
    {
        if (m_registry.hasImageArray(name))
        {
            m_registry.setImageArrayLayout(name, layout);
        }
        else
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

std::vector<std::string> FrameGraph::passNames() const
{
    std::vector<std::string> names;
    names.reserve(m_passes.size());
    for (const auto &p : m_passes)
    {
        names.push_back(p.name);
    }
    return names;
}

std::string FrameGraph::debugDump() const
{
    std::vector<std::vector<framegraph::BarrierDebugInfo>> barriersByPass(m_passes.size());
    for (size_t passIndex = 0; passIndex < m_compiled.size(); ++passIndex)
    {
        auto &output = barriersByPass[passIndex];
        output.reserve(m_compiled[passIndex].imageBarriers.size() +
                       m_compiled[passIndex].bufferBarriers.size());
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

    return framegraph::dumpTopology(m_passes, m_sortedIndices, barriersByPass);
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
