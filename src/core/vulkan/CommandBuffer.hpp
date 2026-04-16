#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

namespace lr
{

// Non-owning wrapper around VkCommandBuffer.
// Passed to pass execute callbacks — the framegraph owns the underlying handle.
class CommandBuffer
{
public:
    explicit CommandBuffer(VkCommandBuffer handle) : m_handle(handle) {}

    // Raw handle — escape hatch for anything not exposed here
    VkCommandBuffer get() const { return m_handle; }

    // Draw calls
    void draw(uint32_t vertexCount,
              uint32_t instanceCount = 1,
              uint32_t firstVertex = 0,
              uint32_t firstInstance = 0);

    void drawIndexed(uint32_t indexCount,
                     uint32_t instanceCount = 1,
                     uint32_t firstIndex = 0,
                     int32_t vertexOffset = 0,
                     uint32_t firstInstance = 0);

    // Compute
    void dispatch(uint32_t x, uint32_t y, uint32_t z = 1);

    // Mesh binding
    void bindVertexBuffer(uint32_t binding, VkBuffer buffer, VkDeviceSize offset = 0);
    void bindIndexBuffer(VkBuffer buffer,
                         VkIndexType indexType = VK_INDEX_TYPE_UINT32,
                         VkDeviceSize offset = 0);

    // Push constants
    template <typename T>
    void pushConstants(VkPipelineLayout layout, VkShaderStageFlags stages, const T &data,
                       uint32_t offset = 0)
    {
        vkCmdPushConstants(m_handle, layout, stages, offset, sizeof(T), &data);
    }

    // Dynamic state
    void setViewport(float x, float y, float width, float height,
                     float minDepth = 0.0f, float maxDepth = 1.0f);
    void setScissor(int32_t offsetX, int32_t offsetY, uint32_t width, uint32_t height);

private:
    VkCommandBuffer m_handle = VK_NULL_HANDLE;
};

}  // namespace lr
