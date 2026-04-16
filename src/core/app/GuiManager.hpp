#pragma once

#include <functional>
#include <string>

namespace lr
{

class ImguiPass;

class GuiManager
{
public:
    explicit GuiManager(ImguiPass &imguiPass);
    ~GuiManager();

    // Set a custom render callback for the GUI
    void setRenderCallback(std::function<void()> callback);

    // Call this at the start of each frame to begin ImGui
    void beginFrame();

    // Call this to render the GUI (executes the render callback)
    void render();

private:
    ImguiPass &m_imguiPass;
    std::function<void()> m_renderCallback;
};

}  // namespace lr
