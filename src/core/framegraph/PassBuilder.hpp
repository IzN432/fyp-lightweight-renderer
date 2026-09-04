#pragma once

#include "Handles.hpp"
#include "core/scene/Mesh.hpp"
#include "core/vulkan/CommandBuffer.hpp"

#include <vulkan/vulkan.h>

#include <functional>
#include <string>
#include <vector>

namespace lr
{

class FrameGraphDefinition;

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
    VkExtent2D         extent     = {0, 0}; // {0,0} = match swapchain size
    VkAttachmentLoadOp loadOp     = VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkClearValue       clearValue = {}; // default: black / depth=0.0
    ImageHandle        image;           // typed frontend identity; name is the compatibility adapter
};

enum class BindingAccess
{
    Read,      // sampled image, uniform buffer — read-only
    Write,     // storage image/buffer written but not read in this pass
    ReadWrite, // storage image/buffer both read and written in this pass
};

// Describes one descriptor binding declared by a pass.
struct BindingDesc
{
    std::string        resourceName;
    uint32_t           binding;
    VkDescriptorType   type; // e.g. COMBINED_IMAGE_SAMPLER, UNIFORM_BUFFER, STORAGE_IMAGE
    uint32_t           descriptorCount = 1;
    VkShaderStageFlags stages          = VK_SHADER_STAGE_ALL_GRAPHICS;
    VkImageLayout      imageLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    BindingAccess      access          = BindingAccess::Read;
    // For STORAGE_IMAGE bindings: which mip level to bind (requires the image
    // to have been registered with mip views, e.g. via registerCubemap with mipLevels > 1).
    uint32_t     mipLevel = 0;
    ImageHandle  image;
    BufferHandle buffer;
};

// Internal description of a pass — populated by PassBuilder, consumed by FrameGraph::compile().
struct PassDesc
{
    std::string name;
    PassHandle  handle;
    PassType    type = PassType::Fullscreen;

    // Shaders
    std::string vertShader;
    std::string fragShader;
    std::string computeShader;

    // Push constants — size in bytes (0 = no push constants).
    uint32_t           pushConstantSize   = 0;
    VkShaderStageFlags pushConstantStages = VK_SHADER_STAGE_COMPUTE_BIT;

    // Primitive topology for Geometry passes — defaults to triangle list.
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Vertex input layout (Geometry passes only) — extracted from GpuMeshLayout at build time
    std::vector<VkVertexInputBindingDescription>   vertexBindings;
    std::vector<VkVertexInputAttributeDescription> vertexAttributes;

    // Vertex and index buffers for Geometry passes. Names are retained as the
    // compatibility bridge to ResourceRegistry during the handle migration.
    struct VertexBufferRef
    {
        uint32_t     binding;
        std::string  bufferName;
        BufferHandle buffer;
    };
    std::vector<VertexBufferRef> vertexBufferRefs;
    std::string                  indexBufferName;
    BufferHandle                 indexBufferHandle;
    uint32_t                     autoIndexCount = 0;

    // Resources
    std::vector<BindingDesc>  bindings; // descriptor bindings (reads + readWrites)
    std::vector<ResourceDesc> writes;   // outputs this pass produces

    // Explicit ordering constraints. Typed handles are primary; names remain
    // available for compatibility with older pass declarations.
    // Use when there is no implicit resource dependency to infer the ordering from
    // (e.g. two passes both writing to the same attachment in sequence).
    std::vector<std::string> explicitDeps;
    std::vector<PassHandle>  explicitDependencyHandles;

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
    PassBuilder(FrameGraphDefinition &definition, PassHandle handle) : m_definition(definition), m_handle(handle) {}

    PassHandle handle() const { return m_handle; }

    // Define the pass type
    PassBuilder &type(PassType t)
    {
        desc().type = t;
        return *this;
    }
    // Set the vertex shader
    PassBuilder &vertShader(std::string path)
    {
        desc().vertShader = std::move(path);
        return *this;
    }
    // Set the fragment shader
    PassBuilder &fragShader(std::string path)
    {
        desc().fragShader = std::move(path);
        return *this;
    }
    // Set the compute shader
    PassBuilder &computeShader(std::string path)
    {
        desc().computeShader = std::move(path);
        return *this;
    }
    // Declare push constants (size in bytes). Stages default to COMPUTE; pass
    // VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT for graphics passes.
    PassBuilder &pushConstantSize(uint32_t size, VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT)
    {
        desc().pushConstantSize   = size;
        desc().pushConstantStages = stages;
        return *this;
    }
    // Set the primitive topology (Geometry passes only). Defaults to triangle list.
    PassBuilder &topology(VkPrimitiveTopology t)
    {
        desc().topology = t;
        return *this;
    }
    // Set the vertex input layout from a GpuMeshLayout (Geometry passes only).
    // Binding and attribute descriptions are extracted immediately so the layout
    // object does not need to outlive the PassBuilder.
    PassBuilder &vertexLayout(const GpuMeshLayout &layout)
    {
        desc().vertexBindings   = layout.bindingDescriptions();
        desc().vertexAttributes = layout.attributeDescriptions();
        return *this;
    }

    // Compatibility string overload. New declarations should use BufferHandle.
    PassBuilder &vertexBuffer(uint32_t binding, std::string name)
    {
        desc().vertexBufferRefs.push_back({.binding = binding, .bufferName = std::move(name)});
        return *this;
    }
    PassBuilder &vertexBuffer(uint32_t binding, BufferHandle buffer)
    {
        desc().vertexBufferRefs.push_back({.binding = binding, .buffer = buffer});
        return *this;
    }
    // Declare the index buffer; the string overload is transitional.
    PassBuilder &indexBuffer(std::string name)
    {
        desc().indexBufferName = std::move(name);
        return *this;
    }
    PassBuilder &indexBuffer(BufferHandle buffer)
    {
        desc().indexBufferHandle = buffer;
        return *this;
    }
    // Store index count so execute callbacks can capture it without separate bookkeeping.
    PassBuilder &indexCount(uint32_t n)
    {
        desc().autoIndexCount = n;
        return *this;
    }
    // Bind resources (descriptors)
    PassBuilder &bind(std::vector<BindingDesc> b)
    {
        desc().bindings = std::move(b);
        return *this;
    }
    // Declare output attachments
    PassBuilder &writes(std::vector<ResourceDesc> descs)
    {
        desc().writes = std::move(descs);
        return *this;
    }
    // Compatibility dependency names plus the typed primary overloads.
    PassBuilder &dependsOn(std::vector<std::string> names)
    {
        desc().explicitDeps = std::move(names);
        return *this;
    }
    PassBuilder &dependsOn(std::vector<PassHandle> handles)
    {
        desc().explicitDependencyHandles = std::move(handles);
        return *this;
    }
    PassBuilder &dependsOn(PassHandle handle)
    {
        desc().explicitDependencyHandles.push_back(handle);
        return *this;
    }
    // Set the execute callback. The pipeline layout is forwarded so push constants
    // can be recorded via cmd.pushConstants(layout, stages, data).
    PassBuilder &execute(std::function<void(CommandBuffer &, VkPipelineLayout)> cb)
    {
        desc().executeCallback = std::move(cb);
        return *this;
    }

private:
    PassDesc &desc();

    FrameGraphDefinition &m_definition;
    PassHandle            m_handle;
};

} // namespace lr
