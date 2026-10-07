#pragma once

#include "RotateGizmo.hpp"
#include "ScaleGizmo.hpp"
#include "TranslateGizmo.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.h>

#include <variant>

namespace lr
{

struct TranslateGizmoRequest
{
    glm::vec3             origin{0.0f};
    TranslateDragHandler *handler = nullptr;
};

struct RotateGizmoRequest
{
    glm::mat4          worldMatrix{1.0f};
    RotateDragHandler *handler = nullptr;
};

struct ScaleGizmoRequest
{
    glm::mat4         worldMatrix{1.0f};
    ScaleDragHandler *handler = nullptr;
};

// A frame contains either no gizmo or exactly one gizmo request. This is the exclusivity boundary:
// editor tools describe what they want to manipulate but cannot show, hide, or rewire shared gizmos.
using GizmoRequest = std::variant<std::monostate, TranslateGizmoRequest, RotateGizmoRequest,
                                  ScaleGizmoRequest>;

class GizmoController
{
public:
    GizmoController(TranslateDragHandler &translation, RotateDragHandler &rotation,
                    ScaleDragHandler &scale)
        : m_translate(translation), m_rotate(rotation), m_scale(scale)
    {}

    void draw(const glm::mat4 &view, const glm::mat4 &projection, bool orthographic,
              VkExtent2D extent, const GizmoRequest &request);

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
