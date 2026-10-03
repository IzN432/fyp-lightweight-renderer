#include "TranslateGizmo.hpp"

#include <imgui.h>
#include <ImGuizmo.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace lr
{

void TranslateGizmo::finishDrag()
{
    if (!m_using)
    {
        return;
    }

    m_dragHandler->endDrag(m_totalDelta);
    m_totalDelta = glm::vec3(0.0f);
    m_using      = false;
}

void TranslateGizmo::draw(const glm::mat4 &view, const glm::mat4 &projection, bool orthographic,
                          VkExtent2D extent, const glm::vec3 &position, bool visible)
{
    if (!visible || extent.width == 0 || extent.height == 0)
    {
        finishDrag();
        m_hovered = false;
        return;
    }

    // The renderer flips projection Y for Vulkan's positive-height viewport. ImGuizmo maps NDC
    // to top-left-origin ImGui coordinates itself, so undo that flip for its screen-space math.
    glm::mat4 gizmoProjection = projection;
    gizmoProjection[1][1] *= -1.0f;

    glm::mat4 model       = glm::translate(glm::mat4(1.0f), position);
    glm::mat4 deltaMatrix = glm::mat4(1.0f);

    ImGuizmo::SetOrthographic(orthographic);
    ImGuizmo::SetRect(0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height));
    ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(gizmoProjection), ImGuizmo::TRANSLATE,
                         ImGuizmo::WORLD, glm::value_ptr(model), glm::value_ptr(deltaMatrix));

    const bool usingNow = ImGuizmo::IsUsing();
    if (usingNow && !m_using)
    {
        m_totalDelta = glm::vec3(0.0f);
        m_dragHandler->beginDrag();
    }

    if (usingNow)
    {
        const glm::vec3 frameDelta(deltaMatrix[3]);
        if (glm::dot(frameDelta, frameDelta) > 0.0f)
        {
            m_dragHandler->translate(frameDelta);
            m_totalDelta += frameDelta;
        }
    } else
    {
        finishDrag();
    }

    m_using   = usingNow;
    m_hovered = ImGuizmo::IsOver(ImGuizmo::TRANSLATE);
}

} // namespace lr
