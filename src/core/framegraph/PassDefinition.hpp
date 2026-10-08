#pragma once

#include "ExtentSpec.hpp"
#include "Handles.hpp"
#include "core/vulkan/ShaderLoader.hpp"

#include <vulkan/vulkan.h>

#include <functional>
#include <limits>
#include <optional>
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
    SampledMultisample,
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
    Index,
    Indirect // draw/dispatch arguments read by vkCmdDraw*Indirect
};

// Applied to every color attachment of a pass.
enum class BlendMode
{
    Opaque,             // no blending
    Alpha,              // src * a + dst * (1 - a)
    PremultipliedAlpha, // src + dst * (1 - a)
    Additive,           // src + dst
};

// Fixed-function state for Geometry/Fullscreen passes. Unset values keep the defaults the engine's
// own passes rely on: back-face culling for Geometry passes (none for Fullscreen), and depth test +
// write exactly when the pass has a depth attachment.
struct GraphicsState
{
    // Number of coverage samples evaluated by rasterization. The public images remain single-sampled:
    // the compiled graph keeps one private multisample image per attachment name, shared by every MSAA
    // pass that renders to it (so a later MSAA pass can LOAD it), and resolves into the public image.
    VkSampleCountFlagBits          samples     = VK_SAMPLE_COUNT_1_BIT;
    BlendMode                      blend       = BlendMode::Opaque;
    VkPolygonMode                  polygonMode = VK_POLYGON_MODE_FILL;
    std::optional<VkCullModeFlags> cullMode;
    VkFrontFace                    frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    std::optional<bool>            depthTest;
    std::optional<bool>            depthWrite;
    VkCompareOp                    depthCompare      = VK_COMPARE_OP_LESS;
    float                          depthBiasConstant = 0.0f;
    float                          depthBiasSlope    = 0.0f;
};

inline constexpr uint32_t noDescriptorBinding = std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t allImageMips        = std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t allImageLayers      = std::numeric_limits<uint32_t>::max();

// Selects a subresource of an image: a mip for a bound descriptor, or a layer for an
// attachment. This affects the bound view only — synchronization is deliberately
// conservative and covers the complete parent image.
struct ImageView
{
    ImageHandle image;
    uint32_t    mipLevel = allImageMips;
    uint32_t    arrayLayer = allImageLayers;

    static ImageView mip(ImageHandle image, uint32_t mipLevel) { return {image, mipLevel, allImageLayers}; }
    static ImageView layer(ImageHandle image, uint32_t arrayLayer) { return {image, allImageMips, arrayLayer}; }
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
    uint32_t           boundLayer      = allImageLayers;

    VkFormat            format  = VK_FORMAT_UNDEFINED;
    ExtentSpec          extent  = ExtentSpec::swapchain();
    VkAttachmentLoadOp  loadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    VkAttachmentStoreOp storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkClearValue        clearValue{};

    bool isDescriptor() const { return binding != noDescriptorBinding; }
    bool isAttachment() const { return usage == ImageUsage::ColorAttachment || usage == ImageUsage::DepthAttachment; }
    // A LOADed depth attachment in a pass with depth writes disabled (see PassBuilder::depth). It is bound
    // in DEPTH_STENCIL_READ_ONLY_OPTIMAL, stored with STORE_OP_NONE and never resolved, so it can be
    // tested against while other passes sample the same image.
    bool isReadOnlyDepth() const { return usage == ImageUsage::DepthAttachment && access == AccessMode::Read; }
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

    ShaderCode                                     vertShader;
    ShaderCode                                     fragShader;
    ShaderCode                                     computeShader;
    uint32_t                                       pushConstantSize   = 0;
    VkShaderStageFlags                             pushConstantStages = VK_SHADER_STAGE_COMPUTE_BIT;
    VkPrimitiveTopology                            topology           = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    std::vector<VkVertexInputBindingDescription>   vertexBindings;
    std::vector<VkVertexInputAttributeDescription> vertexAttributes;
    GraphicsState                                  graphics;

    std::vector<ImageUse>              imageUses;
    std::vector<BufferUse>             bufferUses;
    std::vector<PassHandle>            explicitDependencies;
    std::function<void(PassContext &)> executeCallback;
    // Ordered as if declared after every other pass — see PassBuilder::runsLast().
    bool runsLast = false;
};

} // namespace lr
