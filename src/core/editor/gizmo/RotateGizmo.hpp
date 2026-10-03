#pragma once

#include "core/editor/RotateDragHandler.hpp"

#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.h>

namespace lr
{

class RotateGizmo
{
public:
    explicit RotateGizmo(RotateDragHandler &handler) : m_dragHandler(handler) {}

    void draw(const glm::mat4 &view, const glm::mat4 &projection, bool orthographic, VkExtent2D extent,
              const glm::mat4 &worldMatrix, bool visible);

    bool capturesMouse() const { return m_using || m_hovered; }

private:
    void finishDrag();

    RotateDragHandler &m_dragHandler;
    bool               m_using   = false;
    bool               m_hovered = false;
};

} // namespace lr
