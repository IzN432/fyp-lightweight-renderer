#include "RotateGizmo.hpp"

#include <imgui.h>
#include <ImGuizmo.h>
#include <glm/gtc/type_ptr.hpp>

namespace lr
{

void RotateGizmo::finishDrag()
{
    if (m_using)
    {
        m_dragHandler->endDrag();
        m_using = false;
    }
}

void RotateGizmo::draw(const glm::mat4 &view, const glm::mat4 &projection, bool orthographic,
                       VkExtent2D extent, const glm::mat4 &worldMatrix, bool visible)
{
    if (!visible || extent.width == 0 || extent.height == 0)
    {
        finishDrag();
        m_hovered = false;
        return;
    }

    glm::mat4 gizmoProjection = projection;
    gizmoProjection[1][1] *= -1.0f;
    glm::mat4 manipulatedWorld = worldMatrix;

    ImGuizmo::SetOrthographic(orthographic);
    ImGuizmo::SetRect(0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height));
    const bool changed = ImGuizmo::Manipulate(
        glm::value_ptr(view), glm::value_ptr(gizmoProjection), ImGuizmo::ROTATE, ImGuizmo::LOCAL,
        glm::value_ptr(manipulatedWorld));

    const bool usingNow = ImGuizmo::IsUsing();
    if (usingNow && !m_using)
    {
        m_dragHandler->beginDrag();
    }
    if (changed)
    {
        m_dragHandler->rotateToWorld(manipulatedWorld);
    }
    if (!usingNow)
    {
        finishDrag();
    }

    m_using   = usingNow;
    m_hovered = ImGuizmo::IsOver(ImGuizmo::ROTATE);
}

} // namespace lr
