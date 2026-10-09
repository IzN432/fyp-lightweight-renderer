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
