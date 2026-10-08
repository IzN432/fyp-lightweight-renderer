#include "Viewer.hpp"

#include "core/passes/imgui/ImguiPass.hpp"

#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

#include <optional>
#include <stdexcept>
#include <type_traits>

namespace lr
{

Viewer::Viewer(const Config &config)
{
    m_glfw   = std::make_unique<GlfwContext>();
    m_window = std::make_unique<Window>(Window::Config{
        .width  = config.width,
        .height = config.height,
        .title  = config.title,
    });

    auto extensions = GlfwContext::getRequiredInstanceExtensions();

    m_ctx = std::make_unique<VulkanContext>(VulkanContext::Config{
        .appName                 = config.title,
        .enableValidation        = config.enableValidation,
        .enableDebugNames        = config.enableValidation,
        .extraInstanceExtensions = extensions,
        .extraDeviceExtensions   = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
                                    VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME},
    });

    spdlog::info("Device: {}", m_ctx->getDeviceProperties().properties.deviceName);

    m_allocator = std::make_unique<Allocator>(*m_ctx);
    m_swapchain = std::make_unique<Swapchain>(*m_ctx, *m_window);
    m_resources = std::make_unique<ResourceRegistry>(*m_ctx, *m_allocator, m_swapchain->getExtent());
    m_renderer  = std::make_unique<Renderer>(*m_ctx, *m_swapchain);
    // Before anything registers a dynamic buffer: each gets one copy per frame in flight.
    m_resources->setFramesInFlight(m_renderer->framesInFlight());
    m_fg = std::make_unique<FrameGraph>(*m_ctx, *m_resources);
    if (config.enableGui)
    {
        m_imguiPass = std::make_unique<ImguiPass>(*m_ctx, *m_window, *m_swapchain);
    }

    m_window->setKeyCallback([this](int key, int action) {
        m_input.notifyKey(key, action);
    });
    m_window->setCursorPosCallback([this](double x, double y) {
        m_input.notifyMouseMove(x, y);
    });
    m_window->setMouseButtonCallback([this](int button, int action) {
        m_input.notifyMouseButton(button, action);
    });
    m_window->setScrollCallback([this](double delta) {
        m_input.notifyScroll(delta);
    });

    m_resources->registerExternalImage(kBackbufferName, m_swapchain->getFormat());
    m_backbuffer = m_fg->importBackbuffer(kBackbufferName, m_swapchain->getFormat());
}

CallbackConnection Viewer::onGui(std::function<void()> cb)
{
    if (!m_imguiPass)
    {
        throw std::logic_error("Viewer: onGui() needs a GUI, but this Viewer was created without one");
    }
    return m_guiCallbacks.connect(std::move(cb));
}

template <typename F> auto Viewer::blocking(F &&call)
{
    if (!m_blockingCall)
    {
        return call();
    }
    using Result = decltype(call());
    if constexpr (std::is_void_v<Result>)
    {
        m_blockingCall([&] {
            call();
        });
    } else
    {
        std::optional<Result> result;
        m_blockingCall([&] {
            result.emplace(call());
        });
        return std::move(*result);
    }
}

// Frames may still be executing (e.g. when run() exited with an exception, or never returned
// normally), so wait before any member destroys objects the GPU might be using.
Viewer::~Viewer()
{
    if (m_ctx)
    {
        m_ctx->waitIdleNoThrow();
    }
}

void Viewer::recreateSwapchain()
{
    m_swapchain->recreate();
    m_fg->resize(m_swapchain->getExtent());
    m_frameExecuted = false;
}

void Viewer::addImguiPass()
{
    if (m_imguiPassAdded)
    {
        throw std::logic_error("Viewer: ImGui pass has already been added");
    }

    // runsLast keeps the overlay after every pass touching the swapchain, including passes added
    // after this one (e.g. while running).
    m_fg->addPass("__imgui")
        .type(PassType::Custom)
        .runsLast()
        .colorAttachment(m_backbuffer, m_swapchain->getFormat(), VK_ATTACHMENT_LOAD_OP_LOAD)
        .execute([this](PassContext &ctx) {
            // Without a GUI the pass still declares the window image, so its transitions leave the image
            // ready to present whether or not another pass drew to it.
            if (m_imguiPass)
            {
                m_imguiPass->render(ctx.cmd(), m_swapchain->getImageView(m_currentImageIndex),
                                    ctx.renderingExtent());
            }
        });

    m_imguiPassAdded = true;
}

void Viewer::run()
{
    while (step())
    {
    }
}

bool Viewer::step()
{
    if (m_finished)
    {
        return false;
    }
    if (!m_started)
    {
        if (!m_imguiPassAdded)
        {
            throw std::logic_error("Viewer: addImguiPass() must be called before run()/step()");
        }
        m_fg->compile();
        m_lastFrameTime = glfwGetTime(); // the first frame's dt starts here, not at GLFW init
        m_started       = true;
    }

    if (!m_window->shouldClose())
    {
        try
        {
            renderFrame();
        } catch (...)
        {
            // Leave nothing in flight for whoever handles the exception (e.g. replaces resources).
            m_ctx->waitIdleNoThrow();
            throw;
        }
    }

    if (m_window->shouldClose())
    {
        m_finished = true;
        blocking([&] {
            m_ctx->waitIdle();
        });
        return false;
    }
    return true;
}

void Viewer::renderFrame()
{
    m_window->pollEvents();
    m_input.update();

    if (m_window->wasResized())
    {
        recreateSwapchain();
        m_window->clearResizedFlag();
    }

    if (m_imguiPass)
    {
        m_imguiPass->beginFrame();
        m_guiCallbacks.invoke();
    }

    // Waits for this frame slot's previous submission and acquires the next swapchain image.
    auto [cmd, imageIndex] = blocking([&] {
        return m_renderer->beginFrame(*m_swapchain);
    });
    if (imageIndex == UINT32_MAX)
    {
        recreateSwapchain();
        return;
    }

    m_currentImageIndex = imageIndex;

    // beginFrame() waited for this frame slot's previous submission, so every frame up to
    // `frame - framesInFlight` has finished on the GPU.
    const uint64_t frame          = m_submittedFrames + 1;
    const uint64_t framesInFlight = m_renderer->framesInFlight();
    m_resources->beginFrame(frame, frame > framesInFlight ? frame - framesInFlight : 0);

    const double now = glfwGetTime();
    const float  dt  = static_cast<float>(now - m_lastFrameTime);
    m_lastFrameTime  = now;
    m_updateCallbacks.invoke(dt, m_swapchain->getExtent());
    m_lateUpdateCallbacks.invoke(dt, m_swapchain->getExtent());

    ExternalImageBindings externalImages;
    externalImages.bind(m_backbuffer, m_swapchain->getImage(imageIndex), m_swapchain->getImageView(imageIndex));
    m_fg->execute(cmd, externalImages);
    m_frameExecuted = true;

    const bool presented = blocking([&] {
        return m_renderer->endFrame(*m_swapchain, imageIndex); // submit + present (may wait for vsync)
    });
    ++m_submittedFrames;
    if (!presented)
    {
        recreateSwapchain();
    }
}

} // namespace lr
