#pragma once

#include "Pipeline.hpp"
#include "core/framegraph/PassBuilder.hpp"
#include "core/vulkan/VulkanContext.hpp"

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

namespace lr
{

class GraphicsPipeline : public Pipeline
{
public:
    struct Config
    {
        std::string vertShaderPath;
        std::string fragShaderPath;
        std::vector<VkVertexInputBindingDescription>   vertexBindings;    // empty for Fullscreen passes
        std::vector<VkVertexInputAttributeDescription> vertexAttributes;  // empty for Fullscreen passes
        PassType passType = PassType::Fullscreen;
        std::vector<VkFormat> colorAttachmentFormats;
        VkFormat depthAttachmentFormat = VK_FORMAT_UNDEFINED;
        VkPipelineLayout layout = VK_NULL_HANDLE;  // set by DescriptorAllocator
    };

public:
    GraphicsPipeline(const VulkanContext &ctx, const Config &config);
    ~GraphicsPipeline();

    GraphicsPipeline(const GraphicsPipeline &) = delete;
    GraphicsPipeline &operator=(const GraphicsPipeline &) = delete;

    VkPipeline          get()       const override { return m_pipeline; }
    VkPipelineBindPoint bindPoint() const override { return VK_PIPELINE_BIND_POINT_GRAPHICS; }

private:
    const VulkanContext &m_ctx;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
};

}  // namespace lr
