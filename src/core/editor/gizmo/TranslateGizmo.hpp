#pragma once

#include "core/editor/TranslateDragHandler.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.h>

namespace lr
{

// Immediate-mode translation gizmo backed by ImGuizmo. The target position is supplied every
// frame, while the handler decides whether the resulting world-space delta moves vertices, an
// ARAP handle set, or a scene object.
class TranslateGizmo
{
public:
    explicit TranslateGizmo(TranslateDragHandler &handler) : m_dragHandler(&handler) {}

    void draw(const glm::mat4 &view, const glm::mat4 &projection, bool orthographic, VkExtent2D extent,
              const glm::vec3 &position, bool visible);

    void setDragHandler(TranslateDragHandler &handler) { m_dragHandler = &handler; }
    TranslateDragHandler &dragHandler() const { return *m_dragHandler; }

    bool isInteracting() const { return m_using; }
    bool isHovered() const { return m_hovered; }
    bool capturesMouse() const { return m_using || m_hovered; }

private:
    void finishDrag();

    TranslateDragHandler *m_dragHandler;
    glm::vec3              m_totalDelta{0.0f};
    bool                   m_using   = false;
    bool                   m_hovered = false;
};

} // namespace lr
