#include "GuiManager.hpp"

#include "core/passes/imgui/ImguiPass.hpp"

#include <imgui.h>

namespace lr
{

GuiManager::GuiManager(ImguiPass &imguiPass)
    : m_imguiPass(imguiPass)
    , m_renderCallback([]() { ImGui::ShowDemoWindow(); })
{
}

GuiManager::~GuiManager() = default;

void GuiManager::setRenderCallback(std::function<void()> callback)
{
    m_renderCallback = callback;
}

void GuiManager::beginFrame()
{
    m_imguiPass.beginFrame();
}

void GuiManager::render()
{
    if (m_renderCallback)
        m_renderCallback();
}

}  // namespace lr
