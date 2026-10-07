#pragma once

#include "core/app/InputHandler.hpp"
#include "core/framegraph/FrameGraph.hpp"
#include "core/framegraph/ResourceRegistry.hpp"
#include "core/vulkan/Allocator.hpp"
#include "core/vulkan/Renderer.hpp"
#include "core/vulkan/Swapchain.hpp"
#include "core/vulkan/VulkanContext.hpp"
#include "core/window/Window.hpp"
#include "core/utility/CallbackList.hpp"

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
    // The frame-graph image name of the window's current swapchain image: draw to fg.image(kBackbufferName).
    static constexpr const char *kBackbufferName = "swapchain";

    struct Config
    {
        std::string title            = "Viewer";
        int         width            = 1600;
        int         height           = 900;
        bool        enableValidation = true;
        // Without it there is no ImGui context: onGui() throws, and nothing is drawn over the frame.
        bool enableGui = true;
    };

    // Runs a call that may block on the GPU or the display (fence waits, image acquire, present,
    // waitIdle). See setBlockingCallWrapper.
    using BlockingCallWrapper = std::function<void(const std::function<void()> &call)>;

    explicit Viewer(const Config &config = {});
    ~Viewer();

    Viewer(const Viewer &)            = delete;
    Viewer &operator=(const Viewer &) = delete;

    // -----------------------------------------------------------------------
    // System accessors — use these during app setup before run()
    // -----------------------------------------------------------------------

    FrameGraph          &frameGraph() { return *m_fg; }
    ResourceRegistry    &resources() { return *m_resources; }
    InputHandler        &input() { return m_input; }
    const VulkanContext &context() const { return *m_ctx; }
    Allocator           &allocator() { return *m_allocator; }
    VkFormat             swapchainFormat() const { return m_swapchain->getFormat(); }

    // -----------------------------------------------------------------------
    // Callbacks — set before run()
    // -----------------------------------------------------------------------

    // Called once per frame between imguiPass.beginFrame() and fg.execute().
    // Place all ImGui:: calls here. Throws if the Viewer was created without a GUI.
    CallbackConnection onGui(std::function<void()> cb);
    bool guiEnabled() const { return m_imguiPass != nullptr; }

    // Called once per frame after a valid swapchain image is acquired.
    // dt is seconds since the last frame. extent is the current swapchain size.
    CallbackConnection onUpdate(std::function<void(float dt, VkExtent2D extent)> cb)
    {
        return m_updateCallbacks.connect(std::move(cb));
    }

    // Called once per frame after every onUpdate callback has run, and before the frame graph
    // records/executes GPU work. Use this for end-of-frame bookkeeping that needs to see the
    // results of this frame's updates — e.g. flushing dirty scene state into a single GPU upload
    // rather than reacting to each mutation as it happens.
    CallbackConnection onLateUpdate(std::function<void(float dt, VkExtent2D extent)> cb)
    {
        return m_lateUpdateCallbacks.connect(std::move(cb));
    }

    // Adapters may make a scoped registration last for the Viewer's lifetime when no external
    // owner exists (for example, a Python-owned engine block kept alive by this Viewer).
    void ownConnection(CallbackConnection connection) { m_ownedConnections.push_back(std::move(connection)); }

    // Called for each validation-layer error (only when Config::enableValidation). Replaces any
    // previous handler; see VulkanContext::setValidationErrorHandler.
    void onValidationError(std::function<void(std::string_view message)> cb)
    {
        m_ctx->setValidationErrorHandler(std::move(cb));
    }

    // -----------------------------------------------------------------------
    // Frame loop — run() owns it; step() lets the caller own it instead.
    // -----------------------------------------------------------------------
    // Explicitly append the terminal pass, which composites ImGui (if enabled) over the frame and leaves
    // the window image ready to present, even if no other pass drew to it. Call this after declaring all
    // application passes and before run()/step().
    void addImguiPass();
    bool hasImguiPass() const { return m_imguiPassAdded; }

    // Runs frames until the window is closed: `while (step()) {}`.
    void run();

    // Processes window events and renders one frame (the first call compiles the frame graph first).
    // Returns false once the window has been closed, after waiting for the GPU to finish; from then
    // on it does nothing and keeps returning false. The window only responds while it is being
    // stepped. If a frame throws, the GPU is left idle before the exception propagates.
    bool step();

    // True until the window has been closed (by the user, or requestClose() and the next step()).
    bool isOpen() const { return !m_finished; }

    // Every call that may block on the GPU or the display goes through `wrapper`, which must run it
    // exactly once. Embedders use it to step aside while waiting (e.g. Python releases its GIL, so other
    // threads run while a frame waits for vsync). Nothing that calls back into app code runs inside it.
    void setBlockingCallWrapper(BlockingCallWrapper wrapper) { m_blockingCall = std::move(wrapper); }

    // Ends run() after the current frame finishes (the next step() returns false).
    void requestClose() { m_window->requestClose(); }

    // True once at least one frame has been fully executed since the last
    // swapchain rebuild. Use to guard GPU readbacks that assume images are
    // in their post-render layout.
    bool hasRenderedAtLeastOneFrame() const { return m_frameExecuted; }

private:
    void renderFrame();
    void recreateSwapchain();
    // Runs `call` through the blocking-call wrapper (if any) and returns its result by value.
    template <typename F> auto blocking(F &&call);

    // -----------------------------------------------------------------------
    // Systems — constructed in field order, destroyed in reverse
    // -----------------------------------------------------------------------
    InputHandler                      m_input;
    std::unique_ptr<GlfwContext>      m_glfw;
    std::unique_ptr<Window>           m_window;
    std::unique_ptr<VulkanContext>    m_ctx;
    std::unique_ptr<Allocator>        m_allocator;
    std::unique_ptr<ResourceRegistry> m_resources;
    std::unique_ptr<Swapchain>        m_swapchain;
    std::unique_ptr<Renderer>         m_renderer;
    std::unique_ptr<FrameGraph>       m_fg;
    std::unique_ptr<ImguiPass>        m_imguiPass;
    ImageHandle                       m_backbuffer;

    CallbackList<>                  m_guiCallbacks;
    CallbackList<float, VkExtent2D> m_updateCallbacks;
    CallbackList<float, VkExtent2D> m_lateUpdateCallbacks;
    std::vector<CallbackConnection> m_ownedConnections;
    uint32_t                                            m_currentImageIndex = 0;
    double                                              m_lastFrameTime     = 0.0;
    bool                                                m_frameExecuted     = false;
    bool                                                m_imguiPassAdded    = false;
    bool                                                m_started           = false;
    BlockingCallWrapper                                 m_blockingCall;
    bool                                                m_finished          = false;
    uint64_t                                            m_submittedFrames   = 0;
};

} // namespace lr
