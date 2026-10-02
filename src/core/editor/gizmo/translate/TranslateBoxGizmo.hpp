#pragma once

#include "core/editor/gizmo/Gizmo.hpp"
#include "core/editor/gizmo/TranslateDragHandlerGizmo.hpp"

#include "core/scene/SceneObject.hpp"
#include "core/app/InputHandler.hpp"
#include "core/editor/TranslateDragHandler.hpp"

namespace lr
{

class TranslateBoxGizmo : public Gizmo, public TranslateDragHandlerGizmo
{
public:
    explicit TranslateBoxGizmo(const SceneObject &camera, const InputHandler &input, TranslateDragHandler &handler);
    ~TranslateBoxGizmo() = default;

    void onMouseDown(double ndcX, double ndcY, double aspect) override;
    void onMouseUp(double ndcX, double ndcY, double aspect) override;
    void dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect) override;

    void setDragHandler(TranslateDragHandler &handler) override { m_dragHandler = &handler; }
    TranslateDragHandler &dragHandler() const override { return *m_dragHandler; }

private:
    const SceneObject  &m_camera;
    const InputHandler &m_input;
    TranslateDragHandler *m_dragHandler;

    glm::vec4 m_draggingPlane;
    glm::vec3 m_draggingOrigin;
    glm::vec3 m_currentDraggingOrigin;
};

} // namespace lr
