#pragma once

#include "core/editor/gizmo/Gizmo.hpp"
#include "core/editor/gizmo/DragHandlerGizmo.hpp"

#include "core/scene/SceneObject.hpp"
#include "core/app/InputHandler.hpp"
#include "core/editor/VertexDragHandler.hpp"

namespace lr
{

class TranslateBoxGizmo : public Gizmo, public DragHandlerGizmo
{
public:
    explicit TranslateBoxGizmo(const SceneObject &camera, const InputHandler &input, VertexDragHandler &handler);
    ~TranslateBoxGizmo() = default;

    void onMouseDown(double ndcX, double ndcY, double aspect) override;
    void onMouseUp(double ndcX, double ndcY, double aspect) override;
    void dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect) override;

    void setDragHandler(VertexDragHandler &handler) override { m_dragHandler = &handler; }
    const VertexDragHandler &dragHandler() const override { return *m_dragHandler; }

private:
    const SceneObject  &m_camera;
    const InputHandler &m_input;
    VertexDragHandler *m_dragHandler;

    glm::vec4 m_draggingPlane;
    glm::vec3 m_draggingOrigin;
    glm::vec3 m_currentDraggingOrigin;
};

} // namespace lr
