#pragma once

#include "core/editor/EditorContext.hpp"
#include "core/scene/SceneObjectId.hpp"

#include <functional>
#include <span>

namespace lr
{

class SceneObject;
class SceneObjectDragHandler;
class SceneObjectRotationHandler;
class SceneObjectScaleHandler;

class SceneObjectTransformController
{
public:
    SceneObjectTransformController(SceneObjectDragHandler &translation, SceneObjectRotationHandler &rotation,
                                   SceneObjectScaleHandler &scale);

    void setSelectedTarget(SceneObject *target);
    SceneObject *target() const;

    TransformTool tool() const { return m_tool; }
    void setTool(TransformTool tool) { m_tool = tool; }

    // Selects `tool`, or drops back to no tool if it is already the active one. Gives a keyboard
    // shortcut a way to put the gizmo away again, which the radio buttons cannot express.
    void toggleTool(TransformTool tool) { m_tool = m_tool == tool ? TransformTool::None : tool; }
    void onObjectsDestroyed(std::span<const SceneObjectId> ids);

private:
    void applyTarget(SceneObject *target);

    SceneObjectDragHandler     &m_translation;
    SceneObjectRotationHandler &m_rotation;
    SceneObjectScaleHandler    &m_scale;
    SceneObject                *m_selectedTarget  = nullptr;
    TransformTool               m_tool            = TransformTool::None;
};

} // namespace lr
