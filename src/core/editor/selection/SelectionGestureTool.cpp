#include "SelectionGestureTool.hpp"

#include <glm/geometric.hpp>

namespace lr
{

SelectionGestureTool::SelectionGestureTool(InputHandler &input, SceneObject &camera)
    : SelectionTool(input, camera), m_clickTool(input, camera), m_boxTool(input, camera)
{
    m_clickTool.registerSelectionCallback([this]() {
        m_selectionCallback();
    });
    m_boxTool.registerSelectionCallback([this]() {
        m_selectionCallback();
    });
    m_boxTool.registerHighlightCallback([this]() {
        m_highlightCallback();
    });
}

void SelectionGestureTool::onMouseDown(double ndcX, double ndcY, double aspect)
{
    m_start   = glm::vec2(ndcX, ndcY);
    m_gesture = Gesture::Pending;
}

void SelectionGestureTool::onMouseUp(double ndcX, double ndcY, double aspect)
{
    if (m_gesture == Gesture::Pending)
    {
        const glm::vec2 displacementPixels{
            (static_cast<float>(ndcX) - m_start.x) * 0.5f * static_cast<float>(m_viewportExtent.width),
            (static_cast<float>(ndcY) - m_start.y) * 0.5f * static_cast<float>(m_viewportExtent.height)};
        if (glm::length(displacementPixels) >= kDragThresholdPixels)
        {
            m_gesture = Gesture::Box;
            m_boxTool.onMouseDown(m_start.x, m_start.y, aspect);
            m_boxTool.dragCallback(ndcX, ndcY, 0.0, 0.0, aspect);
        }
    }

    if (m_gesture == Gesture::Box)
    {
        m_boxTool.onMouseUp(ndcX, ndcY, aspect);
    } else if (m_gesture == Gesture::Pending)
    {
        m_gesture = Gesture::Click;
        m_clickTool.onMouseDown(m_start.x, m_start.y, aspect);
        m_clickTool.onMouseUp(ndcX, ndcY, aspect);
    }
    m_gesture = Gesture::None;
}

void SelectionGestureTool::dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect)
{
    if (m_gesture == Gesture::Pending)
    {
        const glm::vec2 displacementPixels{
            (static_cast<float>(ndcX) - m_start.x) * 0.5f * static_cast<float>(m_viewportExtent.width),
            (static_cast<float>(ndcY) - m_start.y) * 0.5f * static_cast<float>(m_viewportExtent.height)};
        if (glm::length(displacementPixels) >= kDragThresholdPixels)
        {
            m_gesture = Gesture::Box;
            m_boxTool.onMouseDown(m_start.x, m_start.y, aspect);
        }
    }

    if (m_gesture == Gesture::Box)
    {
        m_boxTool.dragCallback(ndcX, ndcY, dNdcX, dNdcY, aspect);
    }
}

void SelectionGestureTool::setViewportExtent(VkExtent2D extent)
{
    m_viewportExtent = extent;
    m_clickTool.setViewportExtent(extent);
    m_boxTool.setViewportExtent(extent);
}

void SelectionGestureTool::selectVertices(std::unordered_set<uint32_t> &highlightedVertices,
                                          std::unordered_set<uint32_t> &selectedVertices,
                                          const std::vector<glm::vec3> &vertices)
{
    if (m_gesture == Gesture::Click)
    {
        m_clickTool.selectVertices(highlightedVertices, selectedVertices, vertices);
    } else if (m_gesture == Gesture::Box)
    {
        m_boxTool.selectVertices(highlightedVertices, selectedVertices, vertices);
    }
}

void SelectionGestureTool::highlightVertices(std::unordered_set<uint32_t> &highlightedVertices,
                                             std::unordered_set<uint32_t> &selectedVertices,
                                             const std::vector<glm::vec3> &vertices)
{
    if (m_gesture == Gesture::Box)
    {
        m_boxTool.highlightVertices(highlightedVertices, selectedVertices, vertices);
    }
}

} // namespace lr
