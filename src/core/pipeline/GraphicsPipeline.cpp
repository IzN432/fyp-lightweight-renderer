#include "GraphicsPipeline.hpp"

#include "core/vulkan/ShaderLoader.hpp"
#include "core/vulkan/VkResultUtils.hpp"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace lr
{

GraphicsPipeline::GraphicsPipeline(const VulkanContext &ctx, const Config &config) : m_ctx(ctx)
{
    VkDevice device = ctx.getDevice();

    // -----------------------------------------------------------------------
    // Shaders
    // -----------------------------------------------------------------------

    ShaderModule vertModule(device, config.vertShader);
    ShaderModule fragModule(device, config.fragShader);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule.get();
    stages[0].pName  = "main";

    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule.get();
    stages[1].pName  = "main";

    // -----------------------------------------------------------------------
    // Vertex input — binding and attribute descriptions come pre-built from GpuMeshLayout
    // -----------------------------------------------------------------------

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (config.passType == PassType::Geometry && !config.vertexBindings.empty())
    {
        vertexInput.vertexBindingDescriptionCount   = static_cast<uint32_t>(config.vertexBindings.size());
        vertexInput.pVertexBindingDescriptions      = config.vertexBindings.data();
        vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(config.vertexAttributes.size());
        vertexInput.pVertexAttributeDescriptions    = config.vertexAttributes.data();
    }

    // -----------------------------------------------------------------------
    // Fixed function state
    // -----------------------------------------------------------------------

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = config.topology;

    // Viewport and scissor are dynamic — set per frame via CommandBuffer
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    const GraphicsState &state = config.state;

    VkPipelineRasterizationStateCreateInfo rasterization{};
    rasterization.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.polygonMode = state.polygonMode;
    rasterization.cullMode =
        state.cullMode.value_or(config.passType == PassType::Fullscreen ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);
    rasterization.frontFace               = state.frontFace;
    rasterization.lineWidth               = 1.0f;
    rasterization.depthBiasEnable         = state.depthBiasConstant != 0.0f || state.depthBiasSlope != 0.0f;
    rasterization.depthBiasConstantFactor = state.depthBiasConstant;
    rasterization.depthBiasSlopeFactor    = state.depthBiasSlope;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    bool                                  hasDepth = (config.depthAttachmentFormat != VK_FORMAT_UNDEFINED);
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = state.depthTest.value_or(hasDepth) ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = state.depthWrite.value_or(hasDepth) ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp   = state.depthCompare;

    // One blend attachment per color output, all using the pass's blend mode.
    std::vector<VkPipelineColorBlendAttachmentState> blendAttachments(config.colorAttachmentFormats.size());
    for (auto &blend : blendAttachments)
    {
        blend.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend.blendEnable  = state.blend == BlendMode::Opaque ? VK_FALSE : VK_TRUE;
        blend.colorBlendOp = VK_BLEND_OP_ADD;
        blend.alphaBlendOp = VK_BLEND_OP_ADD;
        switch (state.blend)
        {
            case BlendMode::Opaque:
                break;
            case BlendMode::Alpha:
                blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                break;
            case BlendMode::PremultipliedAlpha:
                blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
                blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                break;
            case BlendMode::Additive:
                blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
                blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
                blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                break;
        }
    }

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = static_cast<uint32_t>(blendAttachments.size());
    colorBlend.pAttachments    = blendAttachments.data();

    VkDynamicState                   dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates    = dynamicStates;

    // -----------------------------------------------------------------------
    // Dynamic rendering — no render pass object needed
    // -----------------------------------------------------------------------

    VkPipelineRenderingCreateInfo renderingCI{};
    renderingCI.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingCI.colorAttachmentCount    = static_cast<uint32_t>(config.colorAttachmentFormats.size());
    renderingCI.pColorAttachmentFormats = config.colorAttachmentFormats.data();
    renderingCI.depthAttachmentFormat   = config.depthAttachmentFormat;

    // -----------------------------------------------------------------------
    // Create pipeline
    // -----------------------------------------------------------------------

    VkGraphicsPipelineCreateInfo pipelineCI{};
    pipelineCI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCI.pNext               = &renderingCI;
    pipelineCI.stageCount          = 2;
    pipelineCI.pStages             = stages;
    pipelineCI.pVertexInputState   = &vertexInput;
    pipelineCI.pInputAssemblyState = &inputAssembly;
    pipelineCI.pViewportState      = &viewportState;
    pipelineCI.pRasterizationState = &rasterization;
    pipelineCI.pMultisampleState   = &multisampling;
    pipelineCI.pDepthStencilState  = &depthStencil;
    pipelineCI.pColorBlendState    = &colorBlend;
    pipelineCI.pDynamicState       = &dynamicState;
    pipelineCI.layout              = config.layout;
    pipelineCI.renderPass          = VK_NULL_HANDLE; // using dynamic rendering

    checkVk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &m_pipeline),
            "GraphicsPipeline: vkCreateGraphicsPipelines");

    spdlog::debug("GraphicsPipeline: created");
}

GraphicsPipeline::~GraphicsPipeline() { vkDestroyPipeline(m_ctx.getDevice(), m_pipeline, nullptr); }

} // namespace lr
