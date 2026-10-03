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

class SceneObjectTransformController : public TransformEditService
{
public:
    SceneObjectTransformController(SceneObjectDragHandler &translation, SceneObjectRotationHandler &rotation,
                                   SceneObjectScaleHandler &scale);

    void setSelectedTarget(SceneObject *target);
    SceneObject *target() const;

    TransformTool tool() const { return m_tool; }
    void setTool(TransformTool tool) { m_tool = tool; }
    bool hasTemporaryEdit() const { return m_temporaryTarget != nullptr; }

    void beginTransformEdit(SceneObject &target, TransformTool tool,
                            std::function<void()> cancel) override;
    void endTransformEdit() override;
    void onObjectsDestroyed(std::span<const SceneObjectId> ids);

private:
    void applyTarget(SceneObject *target);
    void setRecordCommands(bool record);

    SceneObjectDragHandler     &m_translation;
    SceneObjectRotationHandler &m_rotation;
    SceneObjectScaleHandler    &m_scale;
    SceneObject                *m_selectedTarget  = nullptr;
    SceneObject                *m_temporaryTarget = nullptr;
    TransformTool               m_tool            = TransformTool::None;
    std::function<void()>        m_cancelTemporaryEdit;
};

} // namespace lr
