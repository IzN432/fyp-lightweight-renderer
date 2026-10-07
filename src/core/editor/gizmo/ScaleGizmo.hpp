#pragma once

#include "core/editor/ScaleDragHandler.hpp"

#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.h>

namespace lr
{

class ScaleGizmo
{
public:
    explicit ScaleGizmo(ScaleDragHandler &handler) : m_dragHandler(&handler) {}

    void draw(const glm::mat4 &view, const glm::mat4 &projection, bool orthographic, VkExtent2D extent,
              const glm::mat4 &worldMatrix, bool visible);

    void setDragHandler(ScaleDragHandler &handler) { m_dragHandler = &handler; }
    ScaleDragHandler &dragHandler() const { return *m_dragHandler; }

    bool capturesMouse() const { return m_using || m_hovered; }

private:
    void finishDrag();

    ScaleDragHandler *m_dragHandler;
    bool              m_using   = false;
    bool              m_hovered = false;
};

} // namespace lr
