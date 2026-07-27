#include "SelectionManager.hpp"

#include <GLFW/glfw3.h>

namespace lr
{

void SelectionManager::mouseButtonCallback(int button, int action, bool shift, bool ctrl, bool alt)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT)
        return;

    if (action == GLFW_PRESS)
        m_mouseClickedThisFrame = true;
    else if (action == GLFW_RELEASE)
        m_mouseReleasedThisFrame = true;
}

void SelectionManager::updateCallback(float dt, VkExtent2D extent)
{
    if (!m_selectTool)
        return;

    const float aspect = (extent.height == 0)
        ? 1.0f
        : static_cast<float>(extent.width) / static_cast<float>(extent.height);

    double mouseX, mouseY;
    m_input.getMousePos(mouseX, mouseY);
    const float ndcX = (static_cast<float>(mouseX) / extent.width) * 2.0f - 1.0f;
    const float ndcY = (static_cast<float>(mouseY) / extent.height) * 2.0f - 1.0f;

    double dx, dy;
    m_input.getMouseDelta(dx, dy);
    const float dNdcX = (static_cast<float>(dx) / extent.width) * 2.0f;
    const float dNdcY = (static_cast<float>(dy) / extent.height) * 2.0f;

    if (m_mouseClickedThisFrame)
    {
        m_selectTool->onMouseDown(ndcX, ndcY, aspect);
        m_mouseClickedThisFrame = false;
    }
    if (m_mouseReleasedThisFrame)
    {
        m_selectTool->onMouseUp(ndcX, ndcY, aspect);
        m_mouseReleasedThisFrame = false;
    }

    m_selectTool->dragCallback(ndcX, ndcY, dNdcX, dNdcY, aspect);
}

void SelectionManager::setSelectTool(std::unique_ptr<SelectionTool> tool)
{
    m_selectTool = std::move(tool);
    m_selectTool->registerSelectionCallback([this]() {
        if (!m_selectTool)
            return;
        m_selectTool->selectVertices(m_selectedVertices, m_vertices);
        if (m_selectionChangedCallback)
            m_selectionChangedCallback();
    });
}

void SelectionManager::clearSelection()
{
    if (m_selectedVertices.empty())
        return;

    m_selectedVertices.clear();
    if (m_selectionChangedCallback)
        m_selectionChangedCallback();
}

} // namespace lr