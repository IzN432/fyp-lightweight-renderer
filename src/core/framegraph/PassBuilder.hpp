#pragma once

#include "core/scene/Mesh.hpp"
#include "core/vulkan/CommandBuffer.hpp"

#include <vulkan/vulkan.h>

#include <functional>
#include <string>
#include <vector>

namespace lr
{

enum class PassType
{
    Geometry,   // rasterises a mesh — needs vertex/index buffer input
    Fullscreen, // screen-space triangle — no mesh input
    Compute,    // compute shader dispatch
    Custom      // caller owns pipeline, descriptors, and vkCmdBeginRendering/EndRendering;
                // FrameGraph only handles barriers via declared writes()
};

struct ResourceDesc
{
    std::string        name;
    VkFormat           format;
    VkExtent2D         extent     = {0, 0};  // {0,0} = match swapchain size
    VkAttachmentLoadOp loadOp     = VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkClearValue       clearValue = {};      // default: black / depth=0.0
};

enum class BindingAccess
{
    Read,       // sampled image, uniform buffer — read-only
    Write,      // storage image/buffer written but not read in this pass
    ReadWrite,  // storage image/buffer both read and written in this pass
};

// Describes one descriptor binding declared by a pass.
struct BindingDesc
{
    std::string        resourceName;
    uint32_t           binding;
    VkDescriptorType   type;    // e.g. COMBINED_IMAGE_SAMPLER, UNIFORM_BUFFER, STORAGE_IMAGE
    uint32_t           descriptorCount = 1;
    VkShaderStageFlags stages      = VK_SHADER_STAGE_ALL_GRAPHICS;
    VkImageLayout      imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    BindingAccess      access      = BindingAccess::Read;
    // For STORAGE_IMAGE bindings: which mip level to bind (requires the image
    // to have been registered with mip views, e.g. via registerCubemap with mipLevels > 1).
    uint32_t           mipLevel    = 0;
};

// Internal description of a pass — populated by PassBuilder, consumed by FrameGraph::compile().
struct PassDesc
{
    std::string name;
    PassType type = PassType::Fullscreen;

    // Shaders
    std::string vertShader;
    std::string fragShader;
    std::string computeShader;

    // Push constants — size in bytes (0 = no push constants).
    uint32_t           pushConstantSize   = 0;
    VkShaderStageFlags pushConstantStages = VK_SHADER_STAGE_COMPUTE_BIT;

    // Vertex input layout (Geometry passes only) — extracted from GpuMeshLayout at build time
    std::vector<VkVertexInputBindingDescription>   vertexBindings;
    std::vector<VkVertexInputAttributeDescription> vertexAttributes;

    // Vertex and index buffers for Geometry passes — named entries in ResourceRegistry.
    struct VertexBufferRef { uint32_t binding; std::string bufferName; };
    std::vector<VertexBufferRef> vertexBufferRefs;
    std::string                  indexBufferName;
    uint32_t                     autoIndexCount = 0;

    // Resources
    std::vector<BindingDesc>  bindings;   // descriptor bindings (reads + readWrites)
    std::vector<ResourceDesc> writes;     // outputs this pass produces

    // Explicit ordering constraints — pass names that must execute before this one.
    // Use when there is no implicit resource dependency to infer the ordering from
    // (e.g. two passes both writing to the same attachment in sequence).
    std::vector<std::string> explicitDeps;

    // Execute callback — records draw/dispatch calls.
    // The pipeline layout is provided so push constants can be set without
    // needing to hold a raw VkPipelineLayout in app code.
    std::function<void(CommandBuffer &, VkPipelineLayout)> executeCallback;
};

// ---------------------------------------------------------------------------

// Fluent builder returned by FrameGraph::addPass().
// All methods return *this for chaining.
class PassBuilder
{
public:
    explicit PassBuilder(PassDesc &desc) : m_desc(desc) {}

    // Define the pass type
    PassBuilder &type(PassType t)                               { m_desc.type = t;              return *this; }
    // Set the vertex shader
    PassBuilder &vertShader(std::string path)                   { m_desc.vertShader = std::move(path);   return *this; }
    // Set the fragment shader
    PassBuilder &fragShader(std::string path)                   { m_desc.fragShader = std::move(path);   return *this; }
    // Set the compute shader
    PassBuilder &computeShader(std::string path)                { m_desc.computeShader = std::move(path); return *this; }
    // Declare push constants (size in bytes). Stages default to COMPUTE; pass
    // VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT for graphics passes.
    PassBuilder &pushConstantSize(uint32_t size,
                                  VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT)
    {
        m_desc.pushConstantSize   = size;
        m_desc.pushConstantStages = stages;
        return *this;
    }
    // Set the vertex input layout from a GpuMeshLayout (Geometry passes only).
    // Binding and attribute descriptions are extracted immediately so the layout
    // object does not need to outlive the PassBuilder.
    PassBuilder &vertexLayout(const GpuMeshLayout &layout)
    {
        m_desc.vertexBindings   = layout.bindingDescriptions();
        m_desc.vertexAttributes = layout.attributeDescriptions();
        return *this;
    }

    // Declare one vertex buffer binding. name must match a buffer in ResourceRegistry.
    // Call once per binding slot used by the GpuMeshLayout.
    PassBuilder &vertexBuffer(uint32_t binding, std::string name)
    {
        m_desc.vertexBufferRefs.push_back({binding, std::move(name)});
        return *this;
    }
    // Declare the index buffer. name must match a buffer in ResourceRegistry.
    PassBuilder &indexBuffer(std::string name)   { m_desc.indexBufferName  = std::move(name); return *this; }
    // Store index count so execute callbacks can capture it without separate bookkeeping.
    PassBuilder &indexCount(uint32_t n)          { m_desc.autoIndexCount   = n;               return *this; }
    // Bind resources (descriptors)
    PassBuilder &bind(std::vector<BindingDesc> b)               { m_desc.bindings = std::move(b);       return *this; }
    // Declare output attachments
    PassBuilder &writes(std::vector<ResourceDesc> descs)        { m_desc.writes = std::move(descs);     return *this; }
    // Declare explicit dependencies on other passes by name that aren't inferable via resource usage
    PassBuilder &dependsOn(std::vector<std::string> names)      { m_desc.explicitDeps = std::move(names); return *this; }
    // Set the execute callback. The pipeline layout is forwarded so push constants
    // can be recorded via cmd.pushConstants(layout, stages, data).
    PassBuilder &execute(std::function<void(CommandBuffer &, VkPipelineLayout)> cb)
    {
        m_desc.executeCallback = std::move(cb);
        return *this;
    }

private:
    PassDesc &m_desc;
};

}  // namespace lr
