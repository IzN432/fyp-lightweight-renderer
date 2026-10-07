#pragma once

#include "GizmoRequest.hpp"
#include "RotateGizmo.hpp"
#include "ScaleGizmo.hpp"
#include "TranslateGizmo.hpp"

#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.h>

namespace lr
{

class GizmoController
{
public:
    GizmoController(TranslateDragHandler &translation, RotateDragHandler &rotation, ScaleDragHandler &scale)
        : m_translate(translation), m_rotate(rotation), m_scale(scale)
    {}

    void draw(const glm::mat4 &view, const glm::mat4 &projection, bool orthographic, VkExtent2D extent,
              const GizmoRequest &request);

    bool capturesMouse() const
    {
        return m_translate.capturesMouse() || m_rotate.capturesMouse() || m_scale.capturesMouse();
    }

private:
    TranslateGizmo m_translate;
    RotateGizmo    m_rotate;
    ScaleGizmo     m_scale;
};

} // namespace lr
