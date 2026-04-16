#include "ImguiPass.hpp"

#include <spdlog/spdlog.h>
#include "core/vulkan/Swapchain.hpp"
#include <spdlog/spdlog.h>
#include "core/window/Window.hpp"

#include <spdlog/spdlog.h>
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <imgui_impl_glfw.h>
#include <spdlog/spdlog.h>
#include <imgui_impl_vulkan.h>

#include <spdlog/spdlog.h>
#include <stdexcept>

#include <spdlog/spdlog.h>
namespace lr
{

ImguiPass::ImguiPass(const VulkanContext &ctx, const Window &window,
                     const Swapchain &swapchain, uint32_t framesInFlight)
    : m_ctx(ctx)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window.getHandle(), true);

    VkFormat swapchainFormat = swapchain.getFormat();

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion          = ctx.getApiVersion();
    initInfo.Instance            = ctx.getInstance();
    initInfo.PhysicalDevice      = ctx.getPhysicalDevice();
    initInfo.Device              = ctx.getDevice();
    initInfo.QueueFamily         = static_cast<uint32_t>(ctx.getGraphicsQueueFamily());
    initInfo.Queue               = ctx.getGraphicsQueue();
    initInfo.DescriptorPoolSize  = IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE;
    initInfo.MinImageCount       = framesInFlight;
    initInfo.ImageCount          = framesInFlight;
    initInfo.UseDynamicRendering = true;

    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount    = 1;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchainFormat;

    if (!ImGui_ImplVulkan_Init(&initInfo))
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("ImguiPass: ImGui_ImplVulkan_Init failed");
    }
}

ImguiPass::~ImguiPass()
{
    m_ctx.waitIdle();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImguiPass::beginFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImguiPass::render(CommandBuffer &cmd, VkImageView targetView, VkExtent2D extent)
{
    ImGui::Render();

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView   = targetView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;  // composite on top
    colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea           = {{0, 0}, extent};
    renderingInfo.layerCount           = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;

    vkCmdBeginRendering(cmd.get(), &renderingInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd.get());
    vkCmdEndRendering(cmd.get());
}

}  // namespace lr
