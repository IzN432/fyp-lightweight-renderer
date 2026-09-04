#pragma once

#include "ExtentSpec.hpp"
#include "Handles.hpp"

#include <vulkan/vulkan.h>

#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace lr
{

class CommandBuffer;
class PassContext;

enum class PassType
{
    Geometry,
    Fullscreen,
    Compute,
    Custom
};
enum class AccessMode
{
    Read,
    Write,
    ReadWrite
};
enum class ImageUsage
{
    Sampled,
    SampledDepth,
    SampledArray,
    Storage,
    ColorAttachment,
    DepthAttachment
};
enum class BufferUsage
{
    Uniform,
    Storage,
    Vertex,
    Index
};

inline constexpr uint32_t noDescriptorBinding = std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t allImageMips        = std::numeric_limits<uint32_t>::max();

// Mip selection affects the bound descriptor view only. Synchronization is
// deliberately conservative and covers the complete parent image.
struct ImageView
{
    ImageHandle image;
    uint32_t    mipLevel = allImageMips;

    static ImageView mip(ImageHandle image, uint32_t mipLevel) { return {image, mipLevel}; }
};

struct ImageUse
{
    ImageHandle image;
    ImageUsage  usage  = ImageUsage::Sampled;
    AccessMode  access = AccessMode::Read;

    uint32_t           binding         = noDescriptorBinding;
    uint32_t           descriptorCount = 1;
    VkShaderStageFlags stages          = 0;
    uint32_t           boundMip        = allImageMips;

    VkFormat            format  = VK_FORMAT_UNDEFINED;
    ExtentSpec          extent  = ExtentSpec::swapchain();
    VkAttachmentLoadOp  loadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    VkAttachmentStoreOp storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkClearValue        clearValue{};

    bool isDescriptor() const { return binding != noDescriptorBinding; }
    bool isAttachment() const { return usage == ImageUsage::ColorAttachment || usage == ImageUsage::DepthAttachment; }
};

struct BufferUse
{
    BufferHandle       buffer;
    BufferUsage        usage   = BufferUsage::Uniform;
    AccessMode         access  = AccessMode::Read;
    uint32_t           binding = noDescriptorBinding;
    VkShaderStageFlags stages  = 0;

    bool isDescriptor() const { return usage == BufferUsage::Uniform || usage == BufferUsage::Storage; }
};

struct PassDesc
{
    std::string name;
    PassHandle  handle;
    PassType    type = PassType::Fullscreen;

    std::string                                    vertShader;
    std::string                                    fragShader;
    std::string                                    computeShader;
    uint32_t                                       pushConstantSize   = 0;
    VkShaderStageFlags                             pushConstantStages = VK_SHADER_STAGE_COMPUTE_BIT;
    VkPrimitiveTopology                            topology           = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    std::vector<VkVertexInputBindingDescription>   vertexBindings;
    std::vector<VkVertexInputAttributeDescription> vertexAttributes;

    std::vector<ImageUse>              imageUses;
    std::vector<BufferUse>             bufferUses;
    std::vector<PassHandle>            explicitDependencies;
    std::function<void(PassContext &)> executeCallback;
};

} // namespace lr
