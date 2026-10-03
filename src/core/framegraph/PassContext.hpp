#pragma once

#include "Handles.hpp"

#include <vulkan/vulkan.h>

namespace lr
{

class CommandBuffer;
class FrameGraphDefinition;
class ResourceRegistry;
struct PassDesc;

class PassContext
{
public:
    PassContext(CommandBuffer &cmd, VkPipelineLayout pipelineLayout, VkExtent2D renderingExtent,
                const FrameGraphDefinition &definition, const ResourceRegistry &registry, const PassDesc &pass)
        : m_cmd(cmd), m_pipelineLayout(pipelineLayout), m_renderingExtent(renderingExtent), m_definition(definition),
          m_registry(registry), m_pass(pass)
    {}

    CommandBuffer   &cmd() const { return m_cmd; }
    VkPipelineLayout pipelineLayout() const { return m_pipelineLayout; }
    VkExtent2D       renderingExtent() const { return m_renderingExtent; }
    VkExtent2D       extent(ImageHandle image) const;

    // Indirect draws/dispatches. `buffer` must be declared with PassBuilder::indirectBuffer() (so the
    // frame graph synchronises it with whatever wrote it); the read range is bounds-checked.
    void drawIndirect(BufferHandle buffer, uint32_t drawCount = 1, VkDeviceSize offset = 0,
                      uint32_t stride = sizeof(VkDrawIndirectCommand)) const;
    void drawIndexedIndirect(BufferHandle buffer, uint32_t drawCount = 1, VkDeviceSize offset = 0,
                             uint32_t stride = sizeof(VkDrawIndexedIndirectCommand)) const;
    void dispatchIndirect(BufferHandle buffer, VkDeviceSize offset = 0) const;

    // The pass's declared push-constant range (PassBuilder::pushConstantSize).
    uint32_t           pushConstantSize() const;
    VkShaderStageFlags pushConstantStages() const;

private:
    CommandBuffer              &m_cmd;
    VkPipelineLayout            m_pipelineLayout;
    VkExtent2D                  m_renderingExtent;
    const FrameGraphDefinition &m_definition;
    const ResourceRegistry     &m_registry;
    const PassDesc             &m_pass;

    VkBuffer indirectBuffer(BufferHandle buffer, VkDeviceSize offset, uint32_t count, uint32_t stride,
                            VkDeviceSize commandSize) const;
};

} // namespace lr
