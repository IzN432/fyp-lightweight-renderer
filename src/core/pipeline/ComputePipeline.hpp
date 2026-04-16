#pragma once

#include "Pipeline.hpp"
#include "core/vulkan/VulkanContext.hpp"

#include <vulkan/vulkan.h>

#include <string>

namespace lr
{

class ComputePipeline : public Pipeline
{
public:
    ComputePipeline(const VulkanContext &ctx,
                    const std::string   &shaderPath,
                    VkPipelineLayout     layout);
    ~ComputePipeline() override;

    ComputePipeline(const ComputePipeline &) = delete;
    ComputePipeline &operator=(const ComputePipeline &) = delete;

    VkPipeline          get()       const override { return m_pipeline; }
    VkPipelineBindPoint bindPoint() const override { return VK_PIPELINE_BIND_POINT_COMPUTE; }

private:
    const VulkanContext &m_ctx;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
};

}  // namespace lr
