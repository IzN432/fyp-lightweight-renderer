#include "FrameGraph.hpp"

#include "core/pipeline/ComputePipeline.hpp"
#include "core/pipeline/GraphicsPipeline.hpp"

#include <spdlog/spdlog.h>

#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace
{

bool isDepthFormat(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT ||
           format == VK_FORMAT_D24_UNORM_S8_UINT ||
           format == VK_FORMAT_D16_UNORM ||
           format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

struct ResourceState
{
    VkImageLayout         layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 stage  = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2        access = VK_ACCESS_2_NONE;
};

VkPipelineStageFlags2 stageFromShaderStages(VkShaderStageFlags stages)
{
    if (stages & VK_SHADER_STAGE_COMPUTE_BIT)
        return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
}

ResourceState stateForBinding(const lr::BindingDesc &b)
{
    VkPipelineStageFlags2 stage = stageFromShaderStages(b.stages);

    switch (b.type)
    {
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        return { b.imageLayout, stage, VK_ACCESS_2_SHADER_READ_BIT };
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
    {
        VkAccessFlags2 access = 0;
        if (b.access != lr::BindingAccess::Write)
            access |= VK_ACCESS_2_SHADER_READ_BIT;
        if (b.access != lr::BindingAccess::Read)
            access |= VK_ACCESS_2_SHADER_WRITE_BIT;
        return { VK_IMAGE_LAYOUT_GENERAL, stage, access };
    }
    default:
        return {};  // buffer binding — no image layout transition needed
    }
}

ResourceState writeStateForFormat(VkFormat format)
{
    if (isDepthFormat(format))
        return { VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT };

    return { VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT };
}

}  // namespace

namespace lr
{

FrameGraph::FrameGraph(const VulkanContext &ctx, Allocator &allocator, VkExtent2D extent)
    : m_ctx(ctx)
    , m_allocator(allocator)
    , m_registry(ctx, allocator, extent)
    , m_descriptorAllocator(ctx.getDevice())
{
    createDefaultSampler();
}

FrameGraph::~FrameGraph()
{
    destroyCompiledPasses();
    vkDestroySampler(m_ctx.getDevice(), m_defaultSampler, nullptr);
}

// Add a pass with the given name and return a builder for configuring it. The pass won't be compiled until compile() is called.
PassBuilder FrameGraph::addPass(const std::string &name)
{
    for (const auto &p : m_passes)
        if (p.name == name)
        {
            spdlog::error("FrameGraph: duplicate pass name '{}'", name);
            throw std::runtime_error("FrameGraph: duplicate pass name '" + name + "'");
        }

    m_passes.push_back(PassDesc{.name = name});
    return PassBuilder(m_passes.back());
}

// Compiles the pass graph: topological sort, resource allocation, pipeline and descriptor set creation, barrier generation.
void FrameGraph::destroyCompiledPasses()
{
    // Pipelines must be explicitly destroyed — owned here via unique_ptr.
    // Descriptor set layouts and pipeline layouts are owned by m_descriptorAllocator;
    // do not destroy them here.
    for (auto &cp : m_compiled)
        cp.pipeline.reset();
}

void FrameGraph::compile()
{
    spdlog::info("FrameGraph: compiling {} pass(es)...", m_passes.size());

    destroyCompiledPasses();
    m_compiled.clear();
    m_compiled.resize(m_passes.size());

    sortPasses();
    allocateResources();
    buildDescriptorSets();
    buildPipelines();
    buildBarriers();

    spdlog::info("FrameGraph: compiled OK");
}

void FrameGraph::execute(CommandBuffer &cmd)
{
    m_registry.flushUploads();

    VkExtent2D extent = m_registry.getExtent();

    for (size_t idx : m_sortedIndices)
    {
        const PassDesc    &pass     = m_passes[idx];
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

        // Barriers - patch external image handles, then submit
        if (!compiled.barriers.empty())
        {
            std::vector<VkImageMemoryBarrier2> patchedBarriers;
            patchedBarriers.reserve(compiled.barriers.size());
            for (const auto &cb : compiled.barriers)
            {
                VkImageMemoryBarrier2 b = cb.barrier;
                auto it = m_externalImages.find(cb.resourceName);
                if (it != m_externalImages.end())
                    b.image = it->second.image;
                patchedBarriers.push_back(b);
            }

            VkDependencyInfo dep{};
            dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = static_cast<uint32_t>(patchedBarriers.size());
            dep.pImageMemoryBarriers    = patchedBarriers.data();
            vkCmdPipelineBarrier2(cmd.get(), &dep);
        }

        // Bind pipeline and descriptor set — common to both compute and graphics
        if (compiled.pipeline)
            vkCmdBindPipeline(cmd.get(), compiled.pipeline->bindPoint(),
                              compiled.pipeline->get());

        if (compiled.descriptorSet != VK_NULL_HANDLE && compiled.pipeline)
            vkCmdBindDescriptorSets(cmd.get(), compiled.pipeline->bindPoint(),
                                    compiled.pipelineLayout, 0, 1,
                                    &compiled.descriptorSet, 0, nullptr);

        // Compute and Custom passes skip dynamic rendering — just invoke callback and move on
        if (pass.type == PassType::Compute || pass.type == PassType::Custom)
        {
            if (pass.executeCallback)
                pass.executeCallback(cmd, compiled.pipelineLayout);
            if (useDebugLabels)
                m_ctx.endDebugLabel(cmd.get());
            continue;
        }

        // Graphics passes: begin rendering, set dynamic state, draw, end rendering
        std::vector<VkRenderingAttachmentInfo> colorAttachments;
        VkRenderingAttachmentInfo depthAttachment{};
        bool hasDepth = false;

        for (const auto &write : pass.writes)
        {
            const AllocatedImage *img = m_registry.getImage(write.name);

            // External images (e.g. swapchain) have their view injected per-frame
            VkImageView view = img->view;
            auto extIt = m_externalImages.find(write.name);
            if (extIt != m_externalImages.end())
                view = extIt->second.view;

            VkRenderingAttachmentInfo ai{};
            ai.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            ai.imageView   = view;
            ai.loadOp      = write.loadOp;
            ai.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
            ai.clearValue  = write.clearValue;

            if (isDepthFormat(write.format))
            {
                ai.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                depthAttachment = ai;
                hasDepth = true;
            }
            else
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

        cmd.setViewport(0.0f, 0.0f,
            static_cast<float>(extent.width),
            static_cast<float>(extent.height));
        cmd.setScissor(0, 0, extent.width, extent.height);

        // Bind vertex buffers declared by this pass
        for (const auto &vbRef : pass.vertexBufferRefs)
        {
            const AllocatedBuffer *vb = m_registry.getBuffer(vbRef.bufferName);
            if (!vb)
            {
                spdlog::error("FrameGraph::execute: pass '{}' references missing vertex buffer '{}'",
                              pass.name, vbRef.bufferName);
                throw std::runtime_error(
                    "FrameGraph::execute: pass '" + pass.name +
                    "' references vertex buffer '" + vbRef.bufferName + "' which was not registered");
            }
            cmd.bindVertexBuffer(vbRef.binding, vb->buffer);
        }

        // Bind index buffer if declared
        if (!pass.indexBufferName.empty())
        {
            const AllocatedBuffer *ib = m_registry.getBuffer(pass.indexBufferName);
            if (!ib)
            {
                spdlog::error("FrameGraph::execute: pass '{}' references missing index buffer '{}'",
                              pass.name, pass.indexBufferName);
                throw std::runtime_error(
                    "FrameGraph::execute: pass '" + pass.name +
                    "' references index buffer '" + pass.indexBufferName + "' which was not registered");
            }
            cmd.bindIndexBuffer(ib->buffer);
        }

        if (pass.executeCallback)
            pass.executeCallback(cmd, compiled.pipelineLayout);

        vkCmdEndRendering(cmd.get());

        if (useDebugLabels)
            m_ctx.endDebugLabel(cmd.get());
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
    const size_t n = m_passes.size();

    // resource name → index of the pass that writes it
    // Covers both graphics color/depth outputs and compute storage writes
    std::unordered_map<std::string, size_t> resourceProducers;
    for (size_t i = 0; i < n; ++i)
    {
        for (const auto &write : m_passes[i].writes)
            resourceProducers[write.name] = i;
        for (const auto &b : m_passes[i].bindings)
            if (b.access != BindingAccess::Read)
                resourceProducers[b.resourceName] = i;
    }

    // build a name→index lookup for explicit dependency resolution
    std::unordered_map<std::string, size_t> passIndex;
    for (size_t i = 0; i < n; ++i)
        passIndex[m_passes[i].name] = i;

    // build unique pass→pass out-edges via the resource map
    // edge i→j means pass i must execute before pass j
    std::vector<std::unordered_set<size_t>> outEdges(n);
    for (size_t j = 0; j < n; ++j)
        for (const auto &b : m_passes[j].bindings)
        {
            // find the pass that produces the resource this pass reads, if any
            auto it = resourceProducers.find(b.resourceName);
            if (it != resourceProducers.end() && it->second != j)
                // add an out edge from the producer to this consumer
                outEdges[it->second].insert(j);
        }

    // inject explicit ordering edges declared via dependsOn()
    for (size_t j = 0; j < n; ++j)
        for (const auto &depName : m_passes[j].explicitDeps)
        {
            auto it = passIndex.find(depName);
            if (it == passIndex.end())
            {
                spdlog::error("FrameGraph: pass '{}' depends on unknown pass '{}'",
                              m_passes[j].name, depName);
                throw std::runtime_error(
                    "FrameGraph: pass '" + m_passes[j].name +
                    "' declares dependsOn(\"" + depName + "\") but no such pass exists");
            }
            outEdges[it->second].insert(j);
        }

    // in-degree from the resolved edge sets
    std::vector<int> inDegree(n, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j : outEdges[i])
            ++inDegree[j];

    // seed queue with passes that have no dependencies
    std::queue<size_t> queue;
    for (size_t i = 0; i < n; ++i)
        if (inDegree[i] == 0)
            queue.push(i);

    // Kahn's BFS
    m_sortedIndices.clear();
    while (!queue.empty())
    {
        size_t i = queue.front();
        queue.pop();
        m_sortedIndices.push_back(i);

        for (size_t j : outEdges[i])
            if (--inDegree[j] == 0)
                queue.push(j);
    }

    if (m_sortedIndices.size() != n)
    {
        spdlog::error("FrameGraph: cycle detected in pass dependencies");
        throw std::runtime_error("FrameGraph: cycle detected in pass dependencies");
    }

    spdlog::debug("FrameGraph: pass order:");
    for (size_t idx : m_sortedIndices)
        spdlog::debug("  [{}] {}", idx, m_passes[idx].name);
}

void FrameGraph::allocateResources()
{
    // TODO: for each pass write, register the output image in m_registry
    // Compute usage flags by scanning all reads/writes across passes
    for (const auto &pass : m_passes)
    {
        for (const auto &write : pass.writes)
        {
            if (m_registry.hasImage(write.name))
                continue;

            bool isDepth = isDepthFormat(write.format);

            VkImageUsageFlags usage = isDepth
                ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

            VkImageAspectFlags aspect = isDepth
                ? VK_IMAGE_ASPECT_DEPTH_BIT
                : VK_IMAGE_ASPECT_COLOR_BIT;

            m_registry.registerImage(write.name, write.format, usage, write.extent, aspect);
        }
    }
}

void FrameGraph::buildDescriptorSets()
{
    for (size_t i = 0; i < m_passes.size(); ++i)
    {
        const PassDesc &pass = m_passes[i];
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
            lb.binding = b.binding;
            lb.descriptorType = b.type;
            lb.descriptorCount = b.descriptorCount;
            lb.stageFlags = b.stages;
            layoutBindings.push_back(lb);
        }

        compiled.descriptorLayout =
            m_descriptorAllocator.createLayout(layoutBindings);
        compiled.pipelineLayout =
            m_descriptorAllocator.createPipelineLayout(compiled.descriptorLayout,
                                                        pass.pushConstantSize,
                                                        pass.pushConstantStages);

        if (pass.bindings.empty())
        {
            spdlog::debug("FrameGraph: pass '{}' - no bindings", pass.name);
            continue;
        }

        compiled.descriptorSet = m_descriptorAllocator.allocate(compiled.descriptorLayout);

        for (const auto &b : pass.bindings)
        {
            bool isImage = (b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
                            b.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
                            b.type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);

            if (isImage)
            {
                VkSampler sampler = (b.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                        ? VK_NULL_HANDLE
                                        : m_defaultSampler;

                // Storage images must always be in GENERAL layout
                VkImageLayout layout = (b.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                           ? VK_IMAGE_LAYOUT_GENERAL
                                           : b.imageLayout;

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
                            throw std::runtime_error(
                                "FrameGraph: pass '" + pass.name + "' requests mipLevel=" +
                                std::to_string(b.mipLevel) + " for '" + b.resourceName +
                                "' but image only has " + std::to_string(img->mipViews.size()) +
                                " mip view(s)");
                        return img->mipViews[b.mipLevel];
                    }
                    return img->view;
                };

                if (!images.empty())
                {
                    // It's an image array
                    if (images.size() < b.descriptorCount)
                    {
                        spdlog::error(
                            "FrameGraph: pass '{}' binds image array '{}' with descriptorCount={} "
                            "but only {} slot(s) exist",
                            pass.name, b.resourceName, b.descriptorCount, images.size());
                        throw std::runtime_error(
                            "FrameGraph: pass '" + pass.name + "' binds image array '" +
                            b.resourceName + "' with descriptorCount=" +
                            std::to_string(b.descriptorCount) +
                            " but only " + std::to_string(images.size()) + " slot(s) exist");
                    }

                    std::vector<VkImageView> views;
                    views.reserve(b.descriptorCount);
                    for (uint32_t idx = 0; idx < b.descriptorCount; ++idx)
                    {
                        const AllocatedImage *img = images[idx];
                        if (!img)
                        {
                            spdlog::error(
                                "FrameGraph: pass '{}' binds image array '{}' with empty slot at index {}",
                                pass.name, b.resourceName, idx);
                            throw std::runtime_error(
                                "FrameGraph: pass '" + pass.name + "' binds image array '" +
                                b.resourceName + "' with an empty slot at index " +
                                std::to_string(idx));
                        }
                        views.push_back(pickView(img));
                    }

                    m_descriptorAllocator.writeImageArray(compiled.descriptorSet, b.binding,
                                                          views, sampler, layout, b.type);
                }
                else
                {
                    // Single image
                    const AllocatedImage *img = m_registry.getImage(b.resourceName);
                    if (!img)
                    {
                        spdlog::error(
                            "FrameGraph: pass '{}' binds unknown image '{}'",
                            pass.name, b.resourceName);
                        throw std::runtime_error(
                            "FrameGraph: pass '" + pass.name +
                            "' binds unknown image '" + b.resourceName + "'");
                    }

                    m_descriptorAllocator.writeImage(compiled.descriptorSet, b.binding,
                                                     pickView(img), sampler, layout, b.type);
                }
            }
            else
            {
                if (b.descriptorCount != 1)
                {
                    spdlog::error("FrameGraph: pass '{}' uses unsupported descriptorCount={} for non-image binding '{}'",
                                  pass.name, b.descriptorCount, b.resourceName);
                    throw std::runtime_error(
                        "FrameGraph: pass '" + pass.name + "' uses descriptorCount=" +
                        std::to_string(b.descriptorCount) +
                        " for non-image binding '" + b.resourceName +
                        "', which is not supported yet");
                }

                const AllocatedBuffer *buf = m_registry.getBuffer(b.resourceName);
                if (!buf)
                {
                    spdlog::error("FrameGraph: pass '{}' binds unknown buffer '{}'",
                                  pass.name, b.resourceName);
                    throw std::runtime_error(
                        "FrameGraph: pass '" + pass.name +
                        "' binds unknown buffer '" + b.resourceName + "'");
                }

                m_descriptorAllocator.writeBuffer(compiled.descriptorSet, b.binding,
                                                   buf->buffer, 0, buf->size, b.type);
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
            continue;

        if (pass.type == PassType::Compute)
        {
            if (pass.computeShader.empty())
            {
                spdlog::error("FrameGraph: compute pass '{}' has no shader", pass.name);
                throw std::runtime_error(
                    "FrameGraph: compute pass '" + pass.name + "' has no shader");
            }

            m_compiled[i].pipeline = std::make_unique<ComputePipeline>(
                m_ctx, pass.computeShader, m_compiled[i].pipelineLayout);
            spdlog::debug("FrameGraph: pass '{}' - compute pipeline built", pass.name);
            continue;
        }

        if (pass.vertShader.empty() || pass.fragShader.empty())
        {
            spdlog::error("FrameGraph: graphics pass '{}' has missing shaders (vert='{}', frag='{}')",
                          pass.name, pass.vertShader, pass.fragShader);
            throw std::runtime_error(
                "FrameGraph: graphics pass '" + pass.name + "' has no shaders");
        }

        // Derive attachment formats from declared writes
        std::vector<VkFormat> colorFormats;
        VkFormat depthFormat = VK_FORMAT_UNDEFINED;

        for (const auto &write : pass.writes)
        {
            if (isDepthFormat(write.format))
                depthFormat = write.format;
            else
                colorFormats.push_back(write.format);
        }

        GraphicsPipeline::Config cfg{};
        cfg.vertShaderPath           = pass.vertShader;
        cfg.fragShaderPath           = pass.fragShader;
        cfg.vertexBindings           = pass.vertexBindings;
        cfg.vertexAttributes         = pass.vertexAttributes;
        cfg.passType                 = pass.type;
        cfg.colorAttachmentFormats   = std::move(colorFormats);
        cfg.depthAttachmentFormat    = depthFormat;
        cfg.layout                   = m_compiled[i].pipelineLayout;

        m_compiled[i].pipeline = std::make_unique<GraphicsPipeline>(m_ctx, cfg);
        spdlog::debug("FrameGraph: pass '{}' - pipeline built", pass.name);
    }
}

void FrameGraph::buildBarriers()
{
    std::unordered_map<std::string, ResourceState> resourceStates;

    for (size_t idx : m_sortedIndices)
    {
        const PassDesc &pass     = m_passes[idx];
        CompiledPass   &compiled = m_compiled[idx];

        // For each image binding, emit a barrier if the resource isn't already
        // in the required layout
        for (const auto &b : pass.bindings)
        {
            ResourceState required = stateForBinding(b);
            if (required.layout == VK_IMAGE_LAYOUT_UNDEFINED)
                continue;  // buffer binding - no layout transition

            const AllocatedImage *img = m_registry.getImage(b.resourceName);
            if (!img)
                continue;

            auto it = resourceStates.find(b.resourceName);
            ResourceState current = (it != resourceStates.end()) ? it->second : ResourceState{};

            if (current.layout == required.layout)
                continue;  // already correct, no barrier needed

            VkImageMemoryBarrier2 barrier{};
            barrier.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.srcStageMask  = current.stage;
            barrier.srcAccessMask = current.access;
            barrier.dstStageMask  = required.stage;
            barrier.dstAccessMask = required.access;
            barrier.oldLayout     = current.layout;
            barrier.newLayout     = required.layout;
            barrier.image         = img->image;
            VkImageAspectFlags aspect = isDepthFormat(img->format)
                ? VK_IMAGE_ASPECT_DEPTH_BIT
                : VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange = { aspect, 0, img->mipLevels, 0, img->arrayLayers };

            compiled.barriers.push_back({barrier, b.resourceName});
            resourceStates[b.resourceName] = required;
        }

        // Barriers for write targets - transition into attachment layout before the pass
        for (const auto &write : pass.writes)
        {
            ResourceState required = writeStateForFormat(write.format);

            const AllocatedImage *img = m_registry.getImage(write.name);
            if (!img)
                continue;

            auto it = resourceStates.find(write.name);
            ResourceState current = (it != resourceStates.end()) ? it->second : ResourceState{};

            if (current.layout != required.layout)
            {
                VkImageAspectFlags aspect = isDepthFormat(img->format)
                    ? VK_IMAGE_ASPECT_DEPTH_BIT
                    : VK_IMAGE_ASPECT_COLOR_BIT;

                VkImageMemoryBarrier2 barrier{};
                barrier.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                barrier.srcStageMask  = current.stage;
                barrier.srcAccessMask = current.access;
                barrier.dstStageMask  = required.stage;
                barrier.dstAccessMask = required.access;
                barrier.oldLayout     = current.layout;
                barrier.newLayout     = required.layout;
                barrier.image         = img->image;
                barrier.subresourceRange = { aspect, 0, img->mipLevels, 0, img->arrayLayers };

                compiled.barriers.push_back({barrier, write.name});
            }

            resourceStates[write.name] = required;
        }
    }
}

void FrameGraph::setExternalImage(const std::string &name, VkImage image, VkImageView view)
{
    m_externalImages[name] = {image, view};
}

void FrameGraph::executeOnce()
{
    compile();

    VkDevice device = m_ctx.getDevice();

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.queueFamilyIndex = static_cast<uint32_t>(m_ctx.getGraphicsQueueFamily());
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device, &poolCI, nullptr, &pool) != VK_SUCCESS)
        throw std::runtime_error("FrameGraph::executeOnce: failed to create command pool");

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = pool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer vkCmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &allocInfo, &vkCmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(vkCmd, &beginInfo);

    CommandBuffer cmd(vkCmd);
    execute(cmd);

    vkEndCommandBuffer(vkCmd);

    VkFenceCreateInfo fenceCI{};
    fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(device, &fenceCI, nullptr, &fence);

    VkCommandBufferSubmitInfo cmdSubmit{};
    cmdSubmit.sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmdSubmit.commandBuffer = vkCmd;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos    = &cmdSubmit;

    vkQueueSubmit2(m_ctx.getGraphicsQueue(), 1, &submitInfo, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);  // also frees vkCmd

    // Discard preprocessing passes — registry (images, buffers) stays intact
    destroyCompiledPasses();
    m_compiled.clear();
    m_passes.clear();
    m_sortedIndices.clear();

    spdlog::info("FrameGraph: executeOnce complete, preprocessing passes discarded");
}

std::vector<std::string> FrameGraph::passNames() const
{
    std::vector<std::string> names;
    names.reserve(m_passes.size());
    for (const auto &p : m_passes)
        names.push_back(p.name);
    return names;
}

void FrameGraph::createDefaultSampler()
{
    VkSamplerCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.magFilter = VK_FILTER_LINEAR;
    ci.minFilter = VK_FILTER_LINEAR;
    ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.minLod = 0.0f;
    ci.maxLod = VK_LOD_CLAMP_NONE;

    if (vkCreateSampler(m_ctx.getDevice(), &ci, nullptr, &m_defaultSampler) != VK_SUCCESS)
    {
        spdlog::error("FrameGraph: failed to create default sampler");
        throw std::runtime_error("FrameGraph: failed to create default sampler");
    }
}

}  // namespace lr
