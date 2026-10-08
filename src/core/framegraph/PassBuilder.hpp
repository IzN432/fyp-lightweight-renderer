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
    // In-memory SPIR-V, e.g. from compileGlslFile() (see core/vulkan/ShaderCompiler.hpp); `name` labels it
    // in error messages.
    PassBuilder &vertShader(std::vector<uint32_t> spirv, std::string name = {});
    PassBuilder &fragShader(std::vector<uint32_t> spirv, std::string name = {});
    PassBuilder &computeShader(std::vector<uint32_t> spirv, std::string name = {});
    PassBuilder &pushConstantSize(uint32_t size, VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT);
    PassBuilder &topology(VkPrimitiveTopology topology);
    PassBuilder &vertexLayout(const GpuMeshLayout &layout);
    // Raw layout for vertex data that doesn't come from a Mesh.
    PassBuilder &vertexLayout(std::vector<VkVertexInputBindingDescription>   bindings,
                              std::vector<VkVertexInputAttributeDescription> attributes);

    // Fixed-function state (Geometry/Fullscreen passes) — see GraphicsState for the defaults.
    // Blending reads the attachments, so it also orders this pass after their earlier writers.
    PassBuilder &blend(BlendMode mode);
    PassBuilder &samples(VkSampleCountFlagBits samples);
    PassBuilder &polygonMode(VkPolygonMode mode);
    PassBuilder &cull(VkCullModeFlags mode, VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE);
    PassBuilder &depth(bool test, bool write, VkCompareOp compare = VK_COMPARE_OP_LESS);
    PassBuilder &depthBias(float constant, float slope = 0.0f);
    PassBuilder &renderingLayers(uint32_t count);

    PassBuilder &sampledImage(uint32_t binding, ImageHandle image, VkShaderStageFlags stages,
                              SamplerDesc sampler = {});
    PassBuilder &sampledDepth(uint32_t binding, ImageHandle image, VkShaderStageFlags stages,
                              SamplerDesc sampler = {});
    // Bind the unresolved attachment produced by an earlier MSAA pass (GLSL sampler2DMS).
    PassBuilder &sampledMultisampleImage(uint32_t binding, ImageHandle image, VkShaderStageFlags stages);
    PassBuilder &sampledImageArray(uint32_t binding, ImageHandle images, uint32_t count, VkShaderStageFlags stages,
                                   SamplerDesc sampler = {});
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
    // Arguments for PassContext::drawIndirect/drawIndexedIndirect, e.g. written by a compute pass.
    PassBuilder &indirectBuffer(BufferHandle buffer);

    // The ImageView overloads bind a single array layer of a layered target (see ImageView::layer).
    // The ImageHandle overloads bind the full-range view and delegate to them.
    PassBuilder &colorAttachment(ImageView image, VkFormat format,
                                 VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, VkClearValue clearValue = {},
                                 ExtentSpec extent = ExtentSpec::swapchain());
    PassBuilder &colorAttachment(ImageHandle image, VkFormat format,
                                 VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, VkClearValue clearValue = {},
                                 ExtentSpec extent = ExtentSpec::swapchain());
    PassBuilder &depthAttachment(ImageView image, VkFormat format,
                                 VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, VkClearValue clearValue = {},
                                 ExtentSpec extent = ExtentSpec::swapchain());
    PassBuilder &depthAttachment(ImageHandle image, VkFormat format,
                                 VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, VkClearValue clearValue = {},
                                 ExtentSpec extent = ExtentSpec::swapchain());

    // Order this pass after every other pass that shares a resource with it, including passes
    // declared later — for a final overlay (e.g. the Viewer's ImGui pass).
    PassBuilder &runsLast();

    PassBuilder &dependsOn(PassHandle dependency);
    PassBuilder &dependsOn(std::vector<PassHandle> dependencies);
    PassBuilder &execute(std::function<void(PassContext &)> callback);

private:
    PassDesc    &desc();
    PassBuilder &storageImage(uint32_t binding, ImageView image, VkShaderStageFlags stages, AccessMode access);
    PassBuilder &storageBuffer(uint32_t binding, BufferHandle buffer, VkShaderStageFlags stages, AccessMode access);

    FrameGraphDefinition &m_definition;
    PassHandle            m_handle;
};

} // namespace lr
