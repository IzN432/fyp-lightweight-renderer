#include "SelectionManager.hpp"

#include <GLFW/glfw3.h>
#include <glm/vec4.hpp>

#include <algorithm>

namespace lr
{

namespace
{
// BoxSelectionTool (and any future SelectionTool) hit-tests against camera-space projection, so
// the vertices it sees need to be in the same world space GeometryPass actually renders them in.
std::vector<glm::vec3> toWorldSpace(const std::vector<glm::vec3> &local, const TransformComponent &transform)
{
    const glm::mat4        model = transform.worldMatrix();
    std::vector<glm::vec3> world;
    world.reserve(local.size());
    for (const auto &v : local)
    {
        world.push_back(glm::vec3(model * glm::vec4(v, 1.0f)));
    }
    return world;
}
} // namespace

void SelectionManager::mouseButtonCallback(int button, int action, bool shift, bool ctrl, bool alt)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT)
    {
        return;
    }

    if (action == GLFW_PRESS)
    {
        m_mouseClickedThisFrame = true;
    } else if (action == GLFW_RELEASE)
    {
        m_mouseReleasedThisFrame = true;
    }
}

void SelectionManager::updateCallback(float dt, VkExtent2D extent)
{
    if (!m_selectTool)
    {
        return;
    }

    const float aspect =
        (extent.height == 0) ? 1.0f : static_cast<float>(extent.width) / static_cast<float>(extent.height);
    m_selectTool->setViewportExtent(extent);

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
    m_toolSelectionConnection = m_selectTool->registerSelectionCallback([this]() {
        if (!m_selectTool)
        {
            return;
        }
        m_selectTool->selectVertices(m_highlightedVertices, m_selectedVertices,
                                     toWorldSpace(*m_vertices, *m_meshTransform));
        rebuildColors();
        m_selectionChangedCallbacks.invoke();
        m_colorsChangedCallbacks.invoke();
    });
    m_toolHighlightConnection = m_selectTool->registerHighlightCallback([this]() {
        if (!m_selectTool)
        {
            return;
        }
        m_selectTool->highlightVertices(m_highlightedVertices, m_selectedVertices,
                                        toWorldSpace(*m_vertices, *m_meshTransform));
        rebuildColors();
        m_colorsChangedCallbacks.invoke();
    });
}

void SelectionManager::rebind(const std::vector<glm::vec3> &vertices, const TransformComponent &meshTransform)
{
    m_vertices      = &vertices;
    m_meshTransform = &meshTransform;

    m_selectedVertices.clear();
    m_highlightedVertices.clear();
    m_colors.assign(vertices.size(), kDefaultColor);
    m_roles.assign(vertices.size(), kNoRole);

    m_selectionChangedCallbacks.invoke();
    m_colorsChangedCallbacks.invoke();
    m_roleChangedCallbacks.invoke();
}

void SelectionManager::rebuildColors()
{
    std::fill(m_colors.begin(), m_colors.end(), kDefaultColor);

    for (size_t i = 0; i < m_roles.size(); ++i)
    {
        if (m_roles[i] == kNoRole)
        {
            continue;
        }
        auto it = m_roleColors.find(m_roles[i]);
        if (it != m_roleColors.end())
        {
            m_colors[i] = it->second;
        }
    }

    const glm::vec3 highlightColor = m_selectTool ? m_selectTool->highlightColor() : kDefaultColor;
    for (uint32_t idx : m_highlightedVertices)
    {
        if (idx < m_colors.size())
        {
            m_colors[idx] = highlightColor;
        }
    }
}

void SelectionManager::clearSelection()
{
    // Both lists, since the highlighted-vertex colors (driven by m_colorsChangedCallback) would
    // otherwise stay stale after a clear — selectVertices()/highlightVertices() keep them in sync
    // during normal box-select use, but clearSelection() bypasses that.
    const bool hadSelection = !m_selectedVertices.empty();
    const bool hadHighlight = !m_highlightedVertices.empty();
    if (!hadSelection && !hadHighlight)
    {
        return;
    }

    m_selectedVertices.clear();
    m_highlightedVertices.clear();

    if (hadHighlight)
    {
        rebuildColors();
    }

    if (hadSelection)
    {
        m_selectionChangedCallbacks.invoke();
    }
    if (hadHighlight)
    {
        m_colorsChangedCallbacks.invoke();
    }
}

VertexRoleId SelectionManager::registerRole(const glm::vec3 &color)
{
    const VertexRoleId id = m_nextRoleId++;
    m_roleColors.emplace(id, color);
    return id;
}

void SelectionManager::classifySelectionAs(VertexRoleId role)
{
    if (m_selectedVertices.empty())
    {
        return;
    }

    for (uint32_t idx : m_selectedVertices)
    {
        if (idx < m_roles.size())
        {
            m_roles[idx] = role;
        }
    }

    rebuildColors();
    m_colorsChangedCallbacks.invoke();
    m_roleChangedCallbacks.invoke();
}

void SelectionManager::clearAllRoles()
{
    const bool anyClassified = std::any_of(m_roles.begin(), m_roles.end(), [](VertexRoleId r) {
        return r != kNoRole;
    });
    if (!anyClassified)
    {
        return;
    }

    std::fill(m_roles.begin(), m_roles.end(), kNoRole);

    rebuildColors();
    m_colorsChangedCallbacks.invoke();
    m_roleChangedCallbacks.invoke();
}

VertexRoleId SelectionManager::getVertexRole(uint32_t index) const
{
    return index < m_roles.size() ? m_roles[index] : kNoRole;
}

std::vector<uint32_t> SelectionManager::getIndicesWithRole(VertexRoleId role) const
{
    std::vector<uint32_t> indices;
    for (uint32_t i = 0; i < m_roles.size(); ++i)
    {
        if (m_roles[i] == role)
        {
            indices.push_back(i);
        }
    }
    return indices;
}

} // namespace lr
