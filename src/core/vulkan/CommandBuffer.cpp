#include "CommandBuffer.hpp"

namespace lr
{

void CommandBuffer::draw(uint32_t vertexCount, uint32_t instanceCount,
                          uint32_t firstVertex, uint32_t firstInstance)
{
    vkCmdDraw(m_handle, vertexCount, instanceCount, firstVertex, firstInstance);
}

void CommandBuffer::drawIndexed(uint32_t indexCount, uint32_t instanceCount,
                                 uint32_t firstIndex, int32_t vertexOffset,
                                 uint32_t firstInstance)
{
    vkCmdDrawIndexed(m_handle, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

void CommandBuffer::dispatch(uint32_t x, uint32_t y, uint32_t z)
{
    vkCmdDispatch(m_handle, x, y, z);
}

void CommandBuffer::bindVertexBuffer(uint32_t binding, VkBuffer buffer, VkDeviceSize offset)
{
    vkCmdBindVertexBuffers(m_handle, binding, 1, &buffer, &offset);
}

void CommandBuffer::bindIndexBuffer(VkBuffer buffer, VkIndexType indexType, VkDeviceSize offset)
{
    vkCmdBindIndexBuffer(m_handle, buffer, offset, indexType);
}

void CommandBuffer::setViewport(float x, float y, float width, float height,
                                 float minDepth, float maxDepth)
{
    VkViewport vp{x, y, width, height, minDepth, maxDepth};
    vkCmdSetViewport(m_handle, 0, 1, &vp);
}

void CommandBuffer::setScissor(int32_t offsetX, int32_t offsetY, uint32_t width, uint32_t height)
{
    VkRect2D rect{{offsetX, offsetY}, {width, height}};
    vkCmdSetScissor(m_handle, 0, 1, &rect);
}

}  // namespace lr
