#pragma once

#include "core/editor/gizmo/Gizmo.hpp"
#include "core/editor/gizmo/TranslateDragHandlerGizmo.hpp"

#include "core/scene/SceneObject.hpp"
#include "core/app/InputHandler.hpp"
#include "core/editor/TranslateDragHandler.hpp"

#include <glm/vec3.hpp>

namespace lr
{

enum class TranslateArrowGizmoAxis
{
    X = 0,
    Y = 1,
    Z = 2,
};

class TranslateArrowGizmo : public Gizmo, public TranslateDragHandlerGizmo
{
public:
    explicit TranslateArrowGizmo(TranslateArrowGizmoAxis axis, const SceneObject &camera, const InputHandler &input,
                                 TranslateDragHandler &handler);
    ~TranslateArrowGizmo() = default;

    // Mouse interaction callbacks: override these in derived classes to implement gizmo behavior
    // Mouse positions are in normalized device coordinates (NDC)
    void onMouseDown(double ndcX, double ndcY, double aspect) override;
    void onMouseUp(double ndcX, double ndcY, double aspect) override;
    void dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect) override;

    // Reassigns what this gizmo drives — e.g. swapping from the default vertex-drag behavior onto
    // an ARAP-solve handler once a precompute succeeds.
    void setDragHandler(TranslateDragHandler &handler) override { m_dragHandler = &handler; }
    TranslateDragHandler &dragHandler() const override { return *m_dragHandler; }

private:
    const SceneObject  &m_camera;
    const InputHandler &m_input;
    TranslateDragHandler *m_dragHandler;
    glm::vec3           m_axis;

    glm::vec3 m_currentDraggingOrigin;
    glm::vec3 m_draggingOrigin;
};

} // namespace lr
