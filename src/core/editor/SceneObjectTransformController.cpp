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

void SceneObjectTransformController::setSelectedTarget(SceneObject *target)
{
    m_selectedTarget = target;
    applyTarget(target);
    m_tool = TransformTool::None;
}

SceneObject *SceneObjectTransformController::target() const
{
    return m_selectedTarget;
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
    applyTarget(target());
    if (!target())
    {
        m_tool = TransformTool::None;
    }
}

} // namespace lr
