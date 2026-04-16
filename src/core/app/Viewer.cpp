#include "Viewer.hpp"

#include "core/passes/imgui/ImguiPass.hpp"

#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

namespace lr
{

Viewer::Viewer(const Config &config)
{
    m_glfw     = std::make_unique<GlfwContext>();
    m_window   = std::make_unique<Window>(Window::Config{
        .width    = config.width,
        .height   = config.height,
        .title    = config.title,
    });

    auto extensions = GlfwContext::getRequiredInstanceExtensions();

    m_ctx = std::make_unique<VulkanContext>(VulkanContext::Config{
        .appName                 = config.title,
        .enableValidation        = config.enableValidation,
        .enableDebugNames        = config.enableValidation,
        .extraInstanceExtensions = extensions,
        .extraDeviceExtensions   = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME},
    });

    spdlog::info("Device: {}", m_ctx->getDeviceProperties().properties.deviceName);

    m_allocator = std::make_unique<Allocator>(*m_ctx);
    m_swapchain = std::make_unique<Swapchain>(*m_ctx, *m_window);
    m_renderer  = std::make_unique<Renderer>(*m_ctx, *m_swapchain);
    m_fg        = std::make_unique<FrameGraph>(*m_ctx, *m_allocator, m_swapchain->getExtent());
    m_imguiPass = std::make_unique<ImguiPass>(*m_ctx, *m_window, *m_swapchain);

    m_fg->resources().registerExternalImage("swapchain", m_swapchain->getFormat());
}

Viewer::~Viewer() = default;

void Viewer::recreateSwapchain()
{
    m_swapchain->recreate();
    m_fg->resize(m_swapchain->getExtent());
}

void Viewer::run()
{
    // Snapshot pass names before adding imgui — addPass() inserts into m_passes
    // immediately, so calling passNames() inside the chain would include "__imgui"
    // itself and create a self-cycle.
    auto priorPasses = m_fg->passNames();
    uint32_t currentImageIndex = 0;

    m_fg->addPass("__imgui")
        .type(PassType::Custom)
        .dependsOn(priorPasses)
        .writes({{.name    = "swapchain",
                  .format  = m_swapchain->getFormat(),
                  .loadOp  = VK_ATTACHMENT_LOAD_OP_LOAD}})
        .execute([this, &currentImageIndex](CommandBuffer &cmd, VkPipelineLayout) {
            m_imguiPass->render(cmd,
                                m_swapchain->getImageView(currentImageIndex),
                                m_swapchain->getExtent());
        });

    m_fg->compile();

    while (!m_window->shouldClose())
    {
        m_window->pollEvents();

        if (m_window->wasResized())
        {
            recreateSwapchain();
            m_window->clearResizedFlag();
        }

        m_imguiPass->beginFrame();
        if (m_guiCallback)
            m_guiCallback();

        auto [cmd, imageIndex] = m_renderer->beginFrame(*m_swapchain);
        if (imageIndex == UINT32_MAX)
        {
            recreateSwapchain();
            continue;
        }

        currentImageIndex = imageIndex;

        if (m_updateCallback)
        {
            const double now = glfwGetTime();
            const float  dt  = static_cast<float>(now - m_lastFrameTime);
            m_lastFrameTime  = now;
            m_updateCallback(dt, m_swapchain->getExtent());
        }

        m_fg->setExternalImage("swapchain",
                               m_swapchain->getImage(imageIndex),
                               m_swapchain->getImageView(imageIndex));
        m_fg->execute(cmd);

        Renderer::transitionForPresent(cmd, m_swapchain->getImage(imageIndex));

        if (!m_renderer->endFrame(*m_swapchain, imageIndex))
            recreateSwapchain();
    }

    m_ctx->waitIdle();
}

}  // namespace lr
