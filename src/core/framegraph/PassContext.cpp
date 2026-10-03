#include "PassContext.hpp"

#include "FrameGraphDefinition.hpp"
#include "PassDefinition.hpp"
#include "ResourceRegistry.hpp"
#include "core/vulkan/CommandBuffer.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace lr
{

VkExtent2D PassContext::extent(ImageHandle image) const { return m_registry.getImageExtent(m_definition.name(image)); }

uint32_t PassContext::pushConstantSize() const { return m_pass.pushConstantSize; }

VkShaderStageFlags PassContext::pushConstantStages() const { return m_pass.pushConstantStages; }

VkBuffer PassContext::indirectBuffer(BufferHandle buffer, VkDeviceSize offset, uint32_t count, uint32_t stride,
                                     VkDeviceSize commandSize) const
{
    const bool declared = std::ranges::any_of(m_pass.bufferUses, [&](const BufferUse &use) {
        return use.buffer == buffer && use.usage == BufferUsage::Indirect;
    });
    if (!declared)
    {
        // A handle first created after compile() (e.g. fg.buffer("x") inside this callback) isn't in
        // the compiled graph's snapshot, so its name may not resolve here.
        std::string label = "a buffer";
        try
        {
            label = "'" + m_definition.name(buffer) + "'";
        } catch (const std::exception &)
        {}
        throw std::logic_error("PassContext: pass '" + m_pass.name + "' draws from " + label +
                               " without declaring it with indirectBuffer()");
    }
    const std::string     &name      = m_definition.name(buffer);
    const AllocatedBuffer *allocated = m_registry.getBuffer(name);
    if (!allocated)
    {
        throw std::runtime_error("PassContext: indirect buffer '" + name + "' is not in the registry");
    }
    if (offset % 4 != 0 || (count > 1 && (stride % 4 != 0 || stride < commandSize)))
    {
        throw std::invalid_argument("PassContext: indirect offset must be a multiple of 4, and stride a multiple of 4 "
                                    "no smaller than one command");
    }
    const VkDeviceSize end = count == 0 ? offset : offset + VkDeviceSize(count - 1) * stride + commandSize;
    if (end > allocated->size)
    {
        throw std::out_of_range("PassContext: indirect read of bytes [" + std::to_string(offset) + ", " +
                                std::to_string(end) + ") overruns '" + name + "' (" + std::to_string(allocated->size) +
                                " bytes)");
    }
    return allocated->buffer;
}

void PassContext::drawIndirect(BufferHandle buffer, uint32_t drawCount, VkDeviceSize offset, uint32_t stride) const
{
    m_cmd.drawIndirect(indirectBuffer(buffer, offset, drawCount, stride, sizeof(VkDrawIndirectCommand)), offset,
                       drawCount, stride);
}

void PassContext::drawIndexedIndirect(BufferHandle buffer, uint32_t drawCount, VkDeviceSize offset,
                                      uint32_t stride) const
{
    m_cmd.drawIndexedIndirect(indirectBuffer(buffer, offset, drawCount, stride, sizeof(VkDrawIndexedIndirectCommand)),
                              offset, drawCount, stride);
}

void PassContext::dispatchIndirect(BufferHandle buffer, VkDeviceSize offset) const
{
    m_cmd.dispatchIndirect(indirectBuffer(buffer, offset, 1, 0, sizeof(VkDispatchIndirectCommand)), offset);
}

} // namespace lr
