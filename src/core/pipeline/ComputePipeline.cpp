#include "ComputePipeline.hpp"

#include "core/vulkan/ShaderLoader.hpp"
#include "core/vulkan/VkResultUtils.hpp"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace lr
{

ComputePipeline::ComputePipeline(const VulkanContext &ctx, const ShaderCode &shader, VkPipelineLayout layout)
    : m_ctx(ctx)
{
    VkDevice device = ctx.getDevice();

    ShaderModule shaderModule(device, shader);

    VkPipelineShaderStageCreateInfo stageCI{};
    stageCI.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageCI.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCI.module = shaderModule.get();
    stageCI.pName  = "main";

    VkComputePipelineCreateInfo pipelineCI{};
    pipelineCI.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCI.stage  = stageCI;
    pipelineCI.layout = layout;

    checkVk(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &m_pipeline),
            "ComputePipeline: vkCreateComputePipelines for " + shader.label());

    spdlog::debug("ComputePipeline: created");
}

ComputePipeline::~ComputePipeline() { vkDestroyPipeline(m_ctx.getDevice(), m_pipeline, nullptr); }

} // namespace lr
