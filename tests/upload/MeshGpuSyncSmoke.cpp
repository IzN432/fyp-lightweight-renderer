#include "core/upload/MeshUploader.hpp"
#include "core/vulkan/VkResultUtils.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace lr;

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

// A tiny readback consumer to check the actual device bytes, not just queued stamps.
class Readback
{
public:
    Readback(const VulkanContext &ctx, Allocator &allocator) : m_ctx(ctx), m_allocator(allocator)
    {
        VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        info.queueFamilyIndex = ctx.getGraphicsQueueFamily();
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        checkVk(vkCreateCommandPool(ctx.getDevice(), &info, nullptr, &m_pool), "smoke: create pool");
        try {
            m_buffer = allocator.createBuffer(16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        } catch (...) {
            vkDestroyCommandPool(ctx.getDevice(), m_pool, nullptr);
            throw;
        }
    }

    ~Readback()
    {
        (void)vkDeviceWaitIdle(m_ctx.getDevice());
        m_allocator.destroy(m_buffer);
        vkDestroyCommandPool(m_ctx.getDevice(), m_pool, nullptr);
    }

    std::array<uint32_t, 4> read(const AllocatedBuffer &source)
    {
        checkVk(vkResetCommandPool(m_ctx.getDevice(), m_pool, 0), "smoke: reset pool");
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = m_pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VkCommandBuffer cmd;
        checkVk(vkAllocateCommandBuffers(m_ctx.getDevice(), &allocation, &cmd), "smoke: allocate command");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        checkVk(vkBeginCommandBuffer(cmd, &begin), "smoke: begin");
        VkBufferCopy copy{.size = source.size};
        vkCmdCopyBuffer(cmd, source.buffer, m_buffer.buffer, 1, &copy);
        VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = m_buffer.buffer;
        barrier.size = source.size;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.bufferMemoryBarrierCount = 1;
        dependency.pBufferMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dependency);
        checkVk(vkEndCommandBuffer(cmd), "smoke: end");
        VkCommandBufferSubmitInfo commands{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        commands.commandBuffer = cmd;
        VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &commands;
        checkVk(vkQueueSubmit2(m_ctx.getGraphicsQueue(), 1, &submit, VK_NULL_HANDLE), "smoke: submit");
        m_ctx.waitIdle();
        checkVk(vmaInvalidateAllocation(m_allocator.getHandle(), m_buffer.allocation, 0, source.size), "smoke: invalidate");
        std::array<uint32_t, 4> values{};
        std::memcpy(values.data(), m_buffer.info.pMappedData, source.size);
        vkFreeCommandBuffers(m_ctx.getDevice(), m_pool, 1, &cmd);
        return values;
    }

private:
    const VulkanContext &m_ctx;
    Allocator &m_allocator;
    VkCommandPool m_pool = VK_NULL_HANDLE;
    AllocatedBuffer m_buffer;
};

int main()
try
{
    VulkanContext context({.appName = "Mesh sync smoke", .enableValidation = true});
    Allocator allocator(context);
    ResourceRegistry registry(context, allocator, {1, 1});
    Readback readback(context, allocator);
    std::array<uint32_t, 4> values{1, 2, 3, 4};
    registry.uploadBuffer("copy", values.data(), sizeof(values), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    registry.flushUploads();
    require(readback.read(*registry.getBuffer("copy")) == values, "initial upload bytes");
    values[0] = 9;
    registry.reuploadBuffer("copy", values.data(), sizeof(uint32_t));
    registry.flushUploads();
    require(readback.read(*registry.getBuffer("copy")) == values, "partial reupload must preserve tail");
    // Queue a large old upload, then shrink before flush. The obsolete copy must be discarded.
    registry.reuploadBuffer("copy", values.data(), sizeof(values));
    values[0] = 7;
    registry.replaceUploadedBuffer("copy", values.data(), sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    registry.flushUploads();
    require(readback.read(*registry.getBuffer("copy"))[0] == 7, "replacement bytes");

    Mesh mesh;
    mesh.setTopology({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 2}, {{0, 1, 2}});
    mesh.setPerUniqueVertexArray<glm::vec3>("color", std::vector<glm::vec3>(3, {1, 1, 1}));
    MeshUploader uploader(registry);
    VertexBufferUploadConfig geometry{.vertexBufferName = "geometry", .includePosition = true};
    VertexBufferUploadConfig points{.vertexBufferName = "points", .vertexAttributeNames = {"color"}, .includePosition = true};
    uploader.uploadVertexBuffer({&mesh}, geometry);
    uploader.uploadUniqueVertexBuffer({&mesh}, points);
    registry.flushUploads();
    const VkBuffer original = registry.getBuffer("geometry")->buffer;
    mesh.setPositionAt(0, {2, 0, 0});
    mesh.setPositionAt(0, {3, 0, 0});
    require(uploader.synchronizeVertexBuffer({&mesh}, geometry), "geometry must synchronize");
    require(!uploader.synchronizeVertexBuffer({&mesh}, geometry), "unchanged geometry must not synchronize");
    require(uploader.synchronizeUniqueVertexBuffer({&mesh}, points), "points must independently synchronize");
    registry.flushUploads();
    require(registry.getBuffer("geometry")->buffer == original, "content update must preserve allocation");
    mesh.setTopology({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}, {0, 1, 2, 3}, {{0, 1, 2}, {1, 3, 2}});
    uploader.replaceVertexBuffer({&mesh}, geometry);
    uploader.replaceUniqueVertexBuffer({&mesh}, points);
    registry.flushUploads();
    require(registry.getBuffer("geometry")->size == 4 * sizeof(glm::vec3), "geometry replacement size");
    require(!uploader.synchronizeVertexBuffer({&mesh}, geometry), "replacement must establish fresh stamp");
    context.waitIdle();
    std::cout << "Mesh GPU synchronization smoke passed.\n";
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
