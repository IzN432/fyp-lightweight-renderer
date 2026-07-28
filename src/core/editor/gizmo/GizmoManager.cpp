#include "GizmoManager.hpp"


#include <imgui.h>

namespace lr
{

void GizmoManager::updateCallback(float dt, VkExtent2D extent, bool hasRenderedAtLeastOneFrame, ImageReadback &gizmoReadback, ResourceRegistry &resources)
{
    float aspect = (extent.height == 0)
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

    // ---- Gizmo hover detection (GPU readback of previous frame's picking image) ----
    if (hasRenderedAtLeastOneFrame && !ImGui::GetIO().WantCaptureMouse)
    {
        double mx, my;
        m_input.getMousePos(mx, my);
        const uint32_t px = static_cast<uint32_t>(mx);
        const uint32_t py = static_cast<uint32_t>(my);
        if (px < extent.width && py < extent.height)
        {
            const uint32_t raw = gizmoReadback.readPixel(
                resources,
                overlayGeometryPass.pickingImageName(),
                px, py);
            m_hoveredGizmo = (raw != lr::ImageReadback::kNoData && raw != 0) ? raw : ~0u;
        }
        else
        {
            m_hoveredGizmo = ~0u;
        }
    }
    else
    {
        m_hoveredGizmo = ~0u;
    }

    overlayGeometryPass.setHoveredInstance(m_draggingGizmo != ~0u ? m_draggingGizmo : m_hoveredGizmo);

    // ---- Handle mouse button events for the current dragged gizmo ----
    // Released gizmo is tracked separately from m_draggingGizmo, since
    // mouseButtonCallback() already clears m_draggingGizmo synchronously on
    // release (before this update runs on the next frame).
    if (m_mouseReleasedThisFrame)
    {
        m_gizmos[m_releasedGizmo]->onMouseUp(ndcX, ndcY, aspect);
        m_mouseReleasedThisFrame = false;
        m_releasedGizmo = ~0u;
    }

    if (m_draggingGizmo != ~0u)
    {
        if (m_mouseClickedThisFrame)
        {
            m_gizmos[m_draggingGizmo]->onMouseDown(ndcX, ndcY, aspect);
            m_mouseClickedThisFrame = false;
        }

        // ---- Drag callback for the gizmo currently being dragged ----
        m_gizmos[m_draggingGizmo]->dragCallback(ndcX, ndcY, dNdcX, dNdcY, aspect);
    }
}

void GizmoManager::mouseButtonCallback(int button, int action, bool shift, bool ctrl, bool alt)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT)
        return;
    if (action == GLFW_PRESS)
    {
        m_draggingGizmo = m_hoveredGizmo;
        m_mouseClickedThisFrame = true;
    }
    else if (action == GLFW_RELEASE)
    {
        if (m_draggingGizmo != ~0u)
        {
            m_mouseReleasedThisFrame = true;
            m_releasedGizmo = m_draggingGizmo;
        }
        m_draggingGizmo = ~0u;
    }
}

int GizmoManager::addGizmo(std::unique_ptr<Gizmo> gizmo)
{
    const uint32_t id = m_nextGizmoId++;
    gizmo->setPickingId(id);
    m_gizmos.emplace(id, std::move(gizmo));
    m_gizmoHiddenStates[id] = false;
    return id;
}

std::vector<OverlayInstance> GizmoManager::getVisibleGizmoInstances() const
{
    std::vector<OverlayInstance> instances;
    for (const auto &[id, gizmo] : m_gizmos)
    {
        if (!m_gizmoHiddenStates.at(id))
            instances.push_back(gizmo->getInstance());
    }
    return instances;
}

}