#pragma once

#include "core/editor/gizmo/Gizmo.hpp"

#include "core/scene/SceneObject.hpp"
#include "core/app/InputHandler.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/editor/selection/SelectionManager.hpp"
#include "core/editor/command/CommandManager.hpp"

namespace lr
{

class TranslateBoxGizmo : public Gizmo
{
public:
    explicit TranslateBoxGizmo(const SceneObject &camera, const InputHandler &input, VertexManager &vertexManager,
                               SelectionManager &selectionManager, CommandManager &commandManager);
    ~TranslateBoxGizmo() = default;

    void onMouseDown(double ndcX, double ndcY, double aspect) override;
    void onMouseUp(double ndcX, double ndcY, double aspect) override;
    void dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect) override;

private:
    const SceneObject  &m_camera;
    const InputHandler &m_input;
    VertexManager      &m_vertexManager;
    SelectionManager   &m_selectionManager;
    CommandManager     &m_commandManager;

    glm::vec4 m_draggingPlane;
    glm::vec3 m_draggingOrigin;
    glm::vec3 m_currentDraggingOrigin;
};

} // namespace lr
