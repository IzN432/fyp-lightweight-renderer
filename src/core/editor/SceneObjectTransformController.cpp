#include "SceneObjectTransformController.hpp"

#include "core/editor/SceneObjectDragHandler.hpp"
#include "core/editor/SceneObjectRotationHandler.hpp"
#include "core/editor/SceneObjectScaleHandler.hpp"
#include "core/scene/SceneObject.hpp"

#include <algorithm>
#include <utility>

namespace lr
{

SceneObjectTransformController::SceneObjectTransformController(SceneObjectDragHandler &translation,
                                                               SceneObjectRotationHandler &rotation,
                                                               SceneObjectScaleHandler &scale)
    : m_translation(translation), m_rotation(rotation), m_scale(scale)
{}

void SceneObjectTransformController::applyTarget(SceneObject *target)
{
    m_translation.setTarget(target);
    m_rotation.setTarget(target);
    m_scale.setTarget(target);
}

void SceneObjectTransformController::setRecordCommands(bool record)
{
    m_translation.setRecordCommands(record);
    m_rotation.setRecordCommands(record);
    m_scale.setRecordCommands(record);
}

void SceneObjectTransformController::setSelectedTarget(SceneObject *target)
{
    if (m_temporaryTarget && target != m_selectedTarget && m_cancelTemporaryEdit)
    {
        // Clear first because the callback ends the edit through this controller.
        auto cancel = std::move(m_cancelTemporaryEdit);
        cancel();
    }
    m_selectedTarget = target;
    if (!m_temporaryTarget)
    {
        applyTarget(target);
        m_tool = TransformTool::None;
    }
}

SceneObject *SceneObjectTransformController::target() const
{
    return m_temporaryTarget ? m_temporaryTarget : m_selectedTarget;
}

void SceneObjectTransformController::beginTransformEdit(SceneObject &target, TransformTool tool,
                                                        std::function<void()> cancel)
{
    m_temporaryTarget = &target;
    m_tool            = tool;
    m_cancelTemporaryEdit = std::move(cancel);
    applyTarget(&target);
}

void SceneObjectTransformController::endTransformEdit()
{
    m_temporaryTarget = nullptr;
    m_tool            = TransformTool::None;
    m_cancelTemporaryEdit = {};
    applyTarget(m_selectedTarget);
}

void SceneObjectTransformController::onObjectsDestroyed(std::span<const SceneObjectId> ids)
{
    const auto destroyed = [&](const SceneObject *object) {
        return object && std::ranges::find(ids, object->id()) != ids.end();
    };
    if (destroyed(m_selectedTarget))
    {
        m_selectedTarget = nullptr;
    }
    if (destroyed(m_temporaryTarget))
    {
        auto cancel = std::move(m_cancelTemporaryEdit);
        if (cancel)
        {
            cancel();
        }
        else
        {
            m_temporaryTarget = nullptr;
        }
    }
    applyTarget(target());
    if (!target())
    {
        m_tool = TransformTool::None;
    }
}

} // namespace lr
