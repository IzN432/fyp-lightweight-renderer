#pragma once

#include "core/editor/gizmo/Gizmo.hpp"

#include "core/scene/SceneObject.hpp"
#include "core/app/InputHandler.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/editor/selection/SelectionManager.hpp"

#include <glm/vec3.hpp>

namespace lr
{

enum class TranslateArrowGizmoAxis
{
    X = 0,
    Y = 1,
    Z = 2,
};

class TranslateArrowGizmo : public Gizmo
{
public:
    explicit TranslateArrowGizmo(TranslateArrowGizmoAxis axis, const SceneObject &camera, const InputHandler &input,
                                 VertexManager &vertexManager, SelectionManager &selectionManager);
    ~TranslateArrowGizmo() = default;

    // Mouse interaction callbacks: override these in derived classes to implement gizmo behavior
    // Mouse positions are in normalized device coordinates (NDC)
    void onMouseDown(double ndcX, double ndcY, double aspect) override;
    void onMouseUp(double ndcX, double ndcY, double aspect) override;
    void dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect) override;

private:
    const SceneObject  &m_camera;
    const InputHandler &m_input;
    VertexManager      &m_vertexManager;
    SelectionManager   &m_selectionManager;
    glm::vec3           m_axis;
    
    glm::vec3           m_draggingOrigin;
};

} // namespace lr
