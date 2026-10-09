#include "SceneObjectDragHandler.hpp"

#include "core/editor/command/TranslateSceneObjectCommand.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <glm/mat4x4.hpp>
#include <memory>

namespace lr
{

glm::vec3 SceneObjectDragHandler::worldDeltaToLocal(const glm::vec3 &worldDelta) const
{
    if (!m_target || !m_target->parent())
    {
        return worldDelta;
    }

    const SceneObject &parent = m_target->scene().getSceneObject(*m_target->parent());
    return glm::vec3(glm::inverse(parent.worldMatrix()) * glm::vec4(worldDelta, 0.0f));
}

void SceneObjectDragHandler::beginDrag()
{
    m_accumulatedLocalDelta = glm::vec3(0.0f);
}

void SceneObjectDragHandler::translate(const glm::vec3 &frameDelta)
{
    if (!m_target || !m_target->hasComponent<TransformComponent>())
    {
        return;
    }

    const glm::vec3 localDelta = worldDeltaToLocal(frameDelta);
    TransformComponent &transform = m_target->getComponent<TransformComponent>();
    transform.setPosition(transform.transform().position() + localDelta);
    m_accumulatedLocalDelta += localDelta;
}

void SceneObjectDragHandler::endDrag(const glm::vec3 &)
{
    if (!m_target || !m_target->hasComponent<TransformComponent>() ||
        glm::dot(m_accumulatedLocalDelta, m_accumulatedLocalDelta) == 0.0f)
    {
        return;
    }

    // The drag and whatever its commit sets off — an Auto Key keyframe, in practice — are one
    // gesture, so they are one undo.
    CommandTransaction transaction(m_commandManager);
    m_commandManager.appendCommandWithoutExecuting(
        std::make_unique<TranslateSceneObjectCommand>(*m_target, m_accumulatedLocalDelta));
    if (m_commitCallback) m_commitCallback(*m_target);
}

} // namespace lr
