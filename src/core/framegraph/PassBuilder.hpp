#pragma once

#include "PassDefinition.hpp"

namespace lr
{

class FrameGraphDefinition;
class GpuMeshLayout;

class PassBuilder
{
public:
    PassBuilder(FrameGraphDefinition &definition, PassHandle handle) : m_definition(definition), m_handle(handle) {}

    PassHandle handle() const { return m_handle; }

    PassBuilder &type(PassType type);
    PassBuilder &vertShader(std::string path);
    PassBuilder &fragShader(std::string path);
    PassBuilder &computeShader(std::string path);
    PassBuilder &pushConstantSize(uint32_t size, VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT);
    PassBuilder &topology(VkPrimitiveTopology topology);
    PassBuilder &vertexLayout(const GpuMeshLayout &layout);

    PassBuilder &sampledImage(uint32_t binding, ImageHandle image, VkShaderStageFlags stages);
    PassBuilder &sampledDepth(uint32_t binding, ImageHandle image, VkShaderStageFlags stages);
    PassBuilder &sampledImageArray(uint32_t binding, ImageHandle images, uint32_t count, VkShaderStageFlags stages);
    PassBuilder &storageImageRead(uint32_t binding, ImageView image, VkShaderStageFlags stages);
    PassBuilder &storageImageWrite(uint32_t binding, ImageView image, VkShaderStageFlags stages);
    PassBuilder &storageImageReadWrite(uint32_t binding, ImageView image, VkShaderStageFlags stages);
    PassBuilder &storageImageRead(uint32_t binding, ImageHandle image, VkShaderStageFlags stages)
    {
        return storageImageRead(binding, ImageView{image}, stages);
    }
    PassBuilder &storageImageWrite(uint32_t binding, ImageHandle image, VkShaderStageFlags stages)
    {
        return storageImageWrite(binding, ImageView{image}, stages);
    }
    PassBuilder &storageImageReadWrite(uint32_t binding, ImageHandle image, VkShaderStageFlags stages)
    {
        return storageImageReadWrite(binding, ImageView{image}, stages);
    }

    PassBuilder &uniformBuffer(uint32_t binding, BufferHandle buffer, VkShaderStageFlags stages);
    PassBuilder &storageBufferRead(uint32_t binding, BufferHandle buffer, VkShaderStageFlags stages);
    PassBuilder &storageBufferWrite(uint32_t binding, BufferHandle buffer, VkShaderStageFlags stages);
    PassBuilder &storageBufferReadWrite(uint32_t binding, BufferHandle buffer, VkShaderStageFlags stages);
    PassBuilder &vertexBuffer(uint32_t binding, BufferHandle buffer);
    PassBuilder &indexBuffer(BufferHandle buffer);

    PassBuilder &colorAttachment(ImageHandle image, VkFormat format,
                                 VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, VkClearValue clearValue = {},
                                 VkExtent2D extent = {});
    PassBuilder &depthAttachment(ImageHandle image, VkFormat format,
                                 VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, VkClearValue clearValue = {},
                                 VkExtent2D extent = {});

    PassBuilder &dependsOn(PassHandle dependency);
    PassBuilder &dependsOn(std::vector<PassHandle> dependencies);
    PassBuilder &execute(std::function<void(CommandBuffer &, VkPipelineLayout)> callback);

private:
    PassDesc    &desc();
    PassBuilder &storageImage(uint32_t binding, ImageView image, VkShaderStageFlags stages, AccessMode access);
    PassBuilder &storageBuffer(uint32_t binding, BufferHandle buffer, VkShaderStageFlags stages, AccessMode access);

    FrameGraphDefinition &m_definition;
    PassHandle            m_handle;
};

} // namespace lr
