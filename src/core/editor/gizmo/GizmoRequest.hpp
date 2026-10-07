#pragma once

#include "core/editor/RotateDragHandler.hpp"
#include "core/editor/ScaleDragHandler.hpp"
#include "core/editor/TranslateDragHandler.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

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
// Kept apart from GizmoController so that code which only *produces* requests — editor states,
// feature tools — does not depend on the concrete ImGuizmo-backed gizmo implementations.
using GizmoRequest = std::variant<std::monostate, TranslateGizmoRequest, RotateGizmoRequest, ScaleGizmoRequest>;

} // namespace lr
