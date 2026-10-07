#include "BoxSelectionTool.hpp"

#include "core/scene/Camera.hpp"

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

namespace lr
{

void BoxSelectionTool::onMouseDown(double ndcX, double ndcY, double aspect)
{
    m_isDragging = true;
    m_boxStart   = glm::vec2(ndcX, ndcY);
    m_boxEnd     = m_boxStart;
}

void BoxSelectionTool::onMouseUp(double ndcX, double ndcY, double aspect)
{
    if (!m_isDragging)
    {
        return;
    }

    m_isDragging = false;
    m_boxEnd     = glm::vec2(ndcX, ndcY);

    m_viewProjectionMatrix = m_camera.getComponent<Camera>().viewProjectionMatrix(aspect);

    // Perform selection
    m_selectionCallbacks.invoke();

    m_boxStart = glm::vec2(0.0, 0.0);
    m_boxEnd   = glm::vec2(0.0, 0.0);
}

void BoxSelectionTool::dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect)
{
    if (m_isDragging)
    {
        m_boxEnd               = glm::vec2(ndcX, ndcY);
        m_viewProjectionMatrix = m_camera.getComponent<Camera>().viewProjectionMatrix(aspect);
        m_highlightCallbacks.invoke();
    }
}

void BoxSelectionTool::selectVertices(std::unordered_set<uint32_t> &highlightedVertices,
                                      std::unordered_set<uint32_t> &selectedVertices,
                                      const std::vector<glm::vec3> &vertices)
{
    selectedVertices.clear();

    for (uint32_t i : highlightedVertices)
    {
        if (i < vertices.size())
        {
            selectedVertices.insert(i);
        }
    }
}

void BoxSelectionTool::highlightVertices(std::unordered_set<uint32_t> &highlightedVertices,
                                         std::unordered_set<uint32_t> &selectedVertices,
                                         const std::vector<glm::vec3> &vertices)
{
    highlightedVertices.clear();
    if (m_input.isShiftPressed())
    {
        highlightedVertices = selectedVertices;
    }

    const glm::vec2 lowerLeft(std::min(m_boxStart.x, m_boxEnd.x), std::min(m_boxStart.y, m_boxEnd.y));
    const glm::vec2 upperRight(std::max(m_boxStart.x, m_boxEnd.x), std::max(m_boxStart.y, m_boxEnd.y));

    for (uint32_t i = 0; i < vertices.size(); ++i)
    {
        const glm::vec4 clip = m_viewProjectionMatrix * glm::vec4(vertices[i], 1.0f);
        if (clip.w <= 0.0f)
        {
            continue;
        }
        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
        if (ndc.z < 0.0f || ndc.z > 1.0f)
        {
            continue;
        }

        if (ndc.x >= lowerLeft.x && ndc.x <= upperRight.x && ndc.y >= lowerLeft.y && ndc.y <= upperRight.y)
        {
            highlightedVertices.insert(i);
        }
    }
}
} // namespace lr
