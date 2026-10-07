#include "GizmoController.hpp"

#include <type_traits>

namespace lr
{

void GizmoController::draw(const glm::mat4 &view, const glm::mat4 &projection, bool orthographic,
                           VkExtent2D extent, const GizmoRequest &request)
{
    const bool translateRequested = std::holds_alternative<TranslateGizmoRequest>(request);
    const bool rotateRequested    = std::holds_alternative<RotateGizmoRequest>(request);
    const bool scaleRequested     = std::holds_alternative<ScaleGizmoRequest>(request);

    // Finish the previous interaction before another gizmo or handler is allowed to take over.
    // Hidden draws are also what clear each wrapper's ImGuizmo hover/capture state.
    if (!translateRequested)
    {
        m_translate.draw(view, projection, orthographic, extent, glm::vec3(0.0f), false);
    }
    if (!rotateRequested)
    {
        m_rotate.draw(view, projection, orthographic, extent, glm::mat4(1.0f), false);
    }
    if (!scaleRequested)
    {
        m_scale.draw(view, projection, orthographic, extent, glm::mat4(1.0f), false);
    }

    std::visit(
        [&](const auto &active) {
            using Request = std::decay_t<decltype(active)>;
            if constexpr (std::is_same_v<Request, TranslateGizmoRequest>)
            {
                if (!active.handler)
                {
                    return;
                }
                if (&m_translate.dragHandler() != active.handler)
                {
                    m_translate.draw(view, projection, orthographic, extent, glm::vec3(0.0f), false);
                    m_translate.setDragHandler(*active.handler);
                }
                m_translate.draw(view, projection, orthographic, extent, active.origin, true);
            }
            else if constexpr (std::is_same_v<Request, RotateGizmoRequest>)
            {
                if (!active.handler)
                {
                    return;
                }
                if (&m_rotate.dragHandler() != active.handler)
                {
                    m_rotate.draw(view, projection, orthographic, extent, glm::mat4(1.0f), false);
                    m_rotate.setDragHandler(*active.handler);
                }
                m_rotate.draw(view, projection, orthographic, extent, active.worldMatrix, true);
            }
            else if constexpr (std::is_same_v<Request, ScaleGizmoRequest>)
            {
                if (!active.handler)
                {
                    return;
                }
                if (&m_scale.dragHandler() != active.handler)
                {
                    m_scale.draw(view, projection, orthographic, extent, glm::mat4(1.0f), false);
                    m_scale.setDragHandler(*active.handler);
                }
                m_scale.draw(view, projection, orthographic, extent, active.worldMatrix, true);
            }
        },
        request);
}

} // namespace lr
