#pragma once

#include "core/app/InputHandler.hpp"
#include "core/framegraph/FrameGraph.hpp"
#include "core/framegraph/ResourceRegistry.hpp"
#include "core/vulkan/Allocator.hpp"
#include "core/vulkan/Renderer.hpp"
#include "core/vulkan/Swapchain.hpp"
#include "core/vulkan/VulkanContext.hpp"
#include "core/window/Window.hpp"

#include <vulkan/vulkan.h>

#include <functional>
#include <memory>
#include <string>

namespace lr
{

class ImguiPass;

// Owns all engine systems (GLFW, Vulkan, swapchain, frame graph, ImGui) and
// runs the frame loop. App code sets callbacks, then calls run().
class Viewer
{
public:
    struct Config
    {
        std::string title            = "Viewer";
        int         width            = 1600;
        int         height           = 900;
        bool        enableValidation = true;
    };

    explicit Viewer(const Config &config = {});
    ~Viewer();

    Viewer(const Viewer &)            = delete;
    Viewer &operator=(const Viewer &) = delete;

    // -----------------------------------------------------------------------
    // System accessors — use these during app setup before run()
    // -----------------------------------------------------------------------

    FrameGraph       &frameGraph() { return *m_fg; }
    ResourceRegistry &resources()  { return m_fg->resources(); }
    InputHandler     &input()      { return m_input; }

    // -----------------------------------------------------------------------
    // Callbacks — set before run()
    // -----------------------------------------------------------------------

    // Called once per frame between imguiPass.beginFrame() and fg.execute().
    // Place all ImGui:: calls here.
    void onGui(std::function<void()> cb) { m_guiCallback = std::move(cb); }

    // Called once per frame after a valid swapchain image is acquired.
    // dt is seconds since the last frame. extent is the current swapchain size.
    void onUpdate(std::function<void(float dt, VkExtent2D extent)> cb) { m_updateCallback = std::move(cb); }

    // -----------------------------------------------------------------------
    // Run — compiles the frame graph and enters the event/render loop.
    // Returns when the window is closed.
    // -----------------------------------------------------------------------
    void run();

private:
    void recreateSwapchain();

    // -----------------------------------------------------------------------
    // Systems — constructed in field order, destroyed in reverse
    // -----------------------------------------------------------------------
    InputHandler                   m_input;
    std::unique_ptr<GlfwContext>   m_glfw;
    std::unique_ptr<Window>        m_window;
    std::unique_ptr<VulkanContext> m_ctx;
    std::unique_ptr<Allocator>     m_allocator;
    std::unique_ptr<Swapchain>     m_swapchain;
    std::unique_ptr<Renderer>      m_renderer;
    std::unique_ptr<FrameGraph>    m_fg;
    std::unique_ptr<ImguiPass>     m_imguiPass;

    std::function<void()>              m_guiCallback;
    std::function<void(float, VkExtent2D)> m_updateCallback;
    double m_lastFrameTime = 0.0;
};

}  // namespace lr
