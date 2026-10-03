#include "SceneObjectRotationHandler.hpp"

#include "core/editor/command/RotateSceneObjectCommand.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <memory>

namespace lr
{

void SceneObjectRotationHandler::beginDrag()
{
    if (m_target && m_target->hasComponent<TransformComponent>())
    {
        m_beforeRotation = m_target->getComponent<TransformComponent>().transform().rotation();
    }
}

void SceneObjectRotationHandler::rotateToWorld(const glm::mat4 &worldMatrix)
{
    if (!m_target || !m_target->hasComponent<TransformComponent>())
    {
        return;
    }

    glm::quat parentWorldRotation(1.0f, 0.0f, 0.0f, 0.0f);
    if (m_target->parent())
    {
        parentWorldRotation = m_target->scene().getSceneObject(*m_target->parent()).worldRotation();
    }

    const glm::quat worldRotation = glm::normalize(glm::quat_cast(glm::mat3(worldMatrix)));
    m_target->getComponent<TransformComponent>().setRotation(
        glm::normalize(glm::inverse(parentWorldRotation) * worldRotation));
}

void SceneObjectRotationHandler::endDrag()
{
    if (!m_target || !m_target->hasComponent<TransformComponent>())
    {
        return;
    }

    const glm::quat afterRotation = m_target->getComponent<TransformComponent>().transform().rotation();
    if (glm::abs(glm::dot(m_beforeRotation, afterRotation)) >= 1.0f - 1e-6f)
    {
        return;
    }

    m_commandManager.appendCommandWithoutExecuting(
        std::make_unique<RotateSceneObjectCommand>(*m_target, m_beforeRotation, afterRotation));
}

} // namespace lr
