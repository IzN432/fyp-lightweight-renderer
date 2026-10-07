#include "ClickSelectionTool.hpp"

#include "core/scene/Camera.hpp"

#include <glm/vec4.hpp>

#include <limits>

namespace lr
{

void ClickSelectionTool::onMouseDown(double ndcX, double ndcY, double aspect)
{
    m_clickPosition = glm::vec2(ndcX, ndcY);
}

void ClickSelectionTool::onMouseUp(double ndcX, double ndcY, double aspect)
{
    m_viewProjectionMatrix = m_camera.getComponent<Camera>().viewProjectionMatrix(aspect);
    m_selectionCallbacks.invoke();
}

void ClickSelectionTool::selectVertices(std::unordered_set<uint32_t> &highlightedVertices,
                                        std::unordered_set<uint32_t> &selectedVertices,
                                        const std::vector<glm::vec3> &vertices)
{
    uint32_t closestIndex = std::numeric_limits<uint32_t>::max();
    float    closestDistanceSquared = kPickRadiusPixels * kPickRadiusPixels;

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

        const float dx = (ndc.x - m_clickPosition.x) * 0.5f * static_cast<float>(m_viewportExtent.width);
        const float dy = (ndc.y - m_clickPosition.y) * 0.5f * static_cast<float>(m_viewportExtent.height);
        const float distanceSquared = dx * dx + dy * dy;
        if (distanceSquared < closestDistanceSquared)
        {
            closestDistanceSquared = distanceSquared;
            closestIndex = i;
        }
    }

    if (!m_input.isShiftPressed())
    {
        selectedVertices.clear();
    }

    if (closestIndex != std::numeric_limits<uint32_t>::max())
    {
        const auto selected = selectedVertices.find(closestIndex);
        if (m_input.isShiftPressed() && selected != selectedVertices.end())
        {
            selectedVertices.erase(selected);
        } else
        {
            selectedVertices.insert(closestIndex);
        }
    }

    highlightedVertices = selectedVertices;
}

} // namespace lr
