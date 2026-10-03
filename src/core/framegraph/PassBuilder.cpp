#include "PassBuilder.hpp"

#include "FrameGraphDefinition.hpp"
#include "core/scene/Mesh.hpp"

#include <stdexcept>

namespace
{

void requireShaderStages(VkShaderStageFlags stages, const char *declaration)
{
    if (stages == 0)
    {
        throw std::invalid_argument(std::string("FrameGraph: ") + declaration + " requires a shader stage");
    }
}

} // namespace

namespace lr
{

PassDesc &PassBuilder::desc() { return m_definition.pass(m_handle); }

PassBuilder &PassBuilder::type(PassType value)
{
    desc().type = value;
    return *this;
}
PassBuilder &PassBuilder::vertShader(std::string path)
{
    desc().vertShader = ShaderCode::fromFile(std::move(path));
    return *this;
}
PassBuilder &PassBuilder::vertShader(std::vector<uint32_t> spirv, std::string name)
{
    desc().vertShader = ShaderCode::fromSpirv(std::move(spirv), std::move(name));
    return *this;
}
PassBuilder &PassBuilder::fragShader(std::string path)
{
    desc().fragShader = ShaderCode::fromFile(std::move(path));
    return *this;
}
PassBuilder &PassBuilder::fragShader(std::vector<uint32_t> spirv, std::string name)
{
    desc().fragShader = ShaderCode::fromSpirv(std::move(spirv), std::move(name));
    return *this;
}
PassBuilder &PassBuilder::computeShader(std::string path)
{
    desc().computeShader = ShaderCode::fromFile(std::move(path));
    return *this;
}
PassBuilder &PassBuilder::computeShader(std::vector<uint32_t> spirv, std::string name)
{
    desc().computeShader = ShaderCode::fromSpirv(std::move(spirv), std::move(name));
    return *this;
}
PassBuilder &PassBuilder::pushConstantSize(uint32_t size, VkShaderStageFlags stages)
{
    desc().pushConstantSize   = size;
    desc().pushConstantStages = stages;
    return *this;
}
PassBuilder &PassBuilder::topology(VkPrimitiveTopology value)
{
    desc().topology = value;
    return *this;
}
PassBuilder &PassBuilder::vertexLayout(const GpuMeshLayout &layout)
{
    desc().vertexBindings   = layout.bindingDescriptions();
    desc().vertexAttributes = layout.attributeDescriptions();
    return *this;
}
PassBuilder &PassBuilder::vertexLayout(std::vector<VkVertexInputBindingDescription>   bindings,
                                       std::vector<VkVertexInputAttributeDescription> attributes)
{
    desc().vertexBindings   = std::move(bindings);
    desc().vertexAttributes = std::move(attributes);
    return *this;
}

PassBuilder &PassBuilder::blend(BlendMode mode)
{
    desc().graphics.blend = mode;
    if (mode != BlendMode::Opaque)
    {
        for (ImageUse &use : desc().imageUses)
        {
            if (use.usage == ImageUsage::ColorAttachment)
            {
                use.access = AccessMode::ReadWrite;
            }
        }
    }
    return *this;
}
PassBuilder &PassBuilder::polygonMode(VkPolygonMode mode)
{
    desc().graphics.polygonMode = mode;
    return *this;
}
PassBuilder &PassBuilder::cull(VkCullModeFlags mode, VkFrontFace frontFace)
{
    desc().graphics.cullMode  = mode;
    desc().graphics.frontFace = frontFace;
    return *this;
}
PassBuilder &PassBuilder::depth(bool test, bool write, VkCompareOp compare)
{
    desc().graphics.depthTest    = test;
    desc().graphics.depthWrite   = write;
    desc().graphics.depthCompare = compare;
    return *this;
}
PassBuilder &PassBuilder::depthBias(float constant, float slope)
{
    desc().graphics.depthBiasConstant = constant;
    desc().graphics.depthBiasSlope    = slope;
    return *this;
}

PassBuilder &PassBuilder::sampledImage(uint32_t binding, ImageHandle image, VkShaderStageFlags stages)
{
    requireShaderStages(stages, "sampled image");
    desc().imageUses.push_back({.image   = image,
                                .usage   = ImageUsage::Sampled,
                                .access  = AccessMode::Read,
                                .binding = binding,
                                .stages  = stages});
    return *this;
}

PassBuilder &PassBuilder::sampledDepth(uint32_t binding, ImageHandle image, VkShaderStageFlags stages)
{
    requireShaderStages(stages, "sampled depth image");
    desc().imageUses.push_back({.image   = image,
                                .usage   = ImageUsage::SampledDepth,
                                .access  = AccessMode::Read,
                                .binding = binding,
                                .stages  = stages});
    return *this;
}

PassBuilder &PassBuilder::sampledImageArray(uint32_t binding, ImageHandle images, uint32_t count,
                                            VkShaderStageFlags stages)
{
    requireShaderStages(stages, "sampled image array");
    if (count == 0)
    {
        throw std::invalid_argument("FrameGraph: sampled image array cannot have zero descriptors");
    }
    desc().imageUses.push_back({.image           = images,
                                .usage           = ImageUsage::SampledArray,
                                .access          = AccessMode::Read,
                                .binding         = binding,
                                .descriptorCount = count,
                                .stages          = stages});
    return *this;
}

PassBuilder &PassBuilder::storageImage(uint32_t binding, ImageView image, VkShaderStageFlags stages, AccessMode access)
{
    requireShaderStages(stages, "storage image");
    desc().imageUses.push_back({.image    = image.image,
                                .usage    = ImageUsage::Storage,
                                .access   = access,
                                .binding  = binding,
                                .stages   = stages,
                                .boundMip = image.mipLevel});
    return *this;
}

PassBuilder &PassBuilder::storageImageRead(uint32_t binding, ImageView image, VkShaderStageFlags stages)
{
    return storageImage(binding, image, stages, AccessMode::Read);
}
PassBuilder &PassBuilder::storageImageWrite(uint32_t binding, ImageView image, VkShaderStageFlags stages)
{
    return storageImage(binding, image, stages, AccessMode::Write);
}
PassBuilder &PassBuilder::storageImageReadWrite(uint32_t binding, ImageView image, VkShaderStageFlags stages)
{
    return storageImage(binding, image, stages, AccessMode::ReadWrite);
}

PassBuilder &PassBuilder::uniformBuffer(uint32_t binding, BufferHandle buffer, VkShaderStageFlags stages)
{
    requireShaderStages(stages, "uniform buffer");
    desc().bufferUses.push_back({.buffer  = buffer,
                                 .usage   = BufferUsage::Uniform,
                                 .access  = AccessMode::Read,
                                 .binding = binding,
                                 .stages  = stages});
    return *this;
}

PassBuilder &PassBuilder::storageBuffer(uint32_t binding, BufferHandle buffer, VkShaderStageFlags stages,
                                        AccessMode access)
{
    requireShaderStages(stages, "storage buffer");
    desc().bufferUses.push_back(
        {.buffer = buffer, .usage = BufferUsage::Storage, .access = access, .binding = binding, .stages = stages});
    return *this;
}

PassBuilder &PassBuilder::storageBufferRead(uint32_t binding, BufferHandle buffer, VkShaderStageFlags stages)
{
    return storageBuffer(binding, buffer, stages, AccessMode::Read);
}
PassBuilder &PassBuilder::storageBufferWrite(uint32_t binding, BufferHandle buffer, VkShaderStageFlags stages)
{
    return storageBuffer(binding, buffer, stages, AccessMode::Write);
}
PassBuilder &PassBuilder::storageBufferReadWrite(uint32_t binding, BufferHandle buffer, VkShaderStageFlags stages)
{
    return storageBuffer(binding, buffer, stages, AccessMode::ReadWrite);
}

PassBuilder &PassBuilder::vertexBuffer(uint32_t binding, BufferHandle buffer)
{
    desc().bufferUses.push_back(
        {.buffer = buffer, .usage = BufferUsage::Vertex, .access = AccessMode::Read, .binding = binding});
    return *this;
}

PassBuilder &PassBuilder::indexBuffer(BufferHandle buffer)
{
    desc().bufferUses.push_back({.buffer = buffer, .usage = BufferUsage::Index, .access = AccessMode::Read});
    return *this;
}

PassBuilder &PassBuilder::indirectBuffer(BufferHandle buffer)
{
    desc().bufferUses.push_back({.buffer = buffer, .usage = BufferUsage::Indirect, .access = AccessMode::Read});
    return *this;
}

PassBuilder &PassBuilder::runsLast()
{
    desc().runsLast = true;
    return *this;
}

PassBuilder &PassBuilder::colorAttachment(ImageHandle image, VkFormat format, VkAttachmentLoadOp loadOp,
                                          VkClearValue clearValue, ExtentSpec extent)
{
    desc().imageUses.push_back(
        {.image      = image,
         .usage      = ImageUsage::ColorAttachment,
         .access     = loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || desc().graphics.blend != BlendMode::Opaque
                           ? AccessMode::ReadWrite
                           : AccessMode::Write,
         .format     = format,
         .extent     = extent,
         .loadOp     = loadOp,
         .clearValue = clearValue});
    return *this;
}

PassBuilder &PassBuilder::depthAttachment(ImageHandle image, VkFormat format, VkAttachmentLoadOp loadOp,
                                          VkClearValue clearValue, ExtentSpec extent)
{
    desc().imageUses.push_back(
        {.image      = image,
         .usage      = ImageUsage::DepthAttachment,
         .access     = loadOp == VK_ATTACHMENT_LOAD_OP_LOAD ? AccessMode::ReadWrite : AccessMode::Write,
         .format     = format,
         .extent     = extent,
         .loadOp     = loadOp,
         .clearValue = clearValue});
    return *this;
}

PassBuilder &PassBuilder::dependsOn(PassHandle dependency)
{
    desc().explicitDependencies.push_back(dependency);
    return *this;
}
PassBuilder &PassBuilder::dependsOn(std::vector<PassHandle> dependencies)
{
    desc().explicitDependencies = std::move(dependencies);
    return *this;
}
PassBuilder &PassBuilder::execute(std::function<void(PassContext &)> callback)
{
    desc().executeCallback = std::move(callback);
    return *this;
}

} // namespace lr
