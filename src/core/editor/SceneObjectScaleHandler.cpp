#include "SceneObjectScaleHandler.hpp"

#include "core/editor/command/ScaleSceneObjectCommand.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>

namespace lr
{

void SceneObjectScaleHandler::beginDrag()
{
    if (m_target && m_target->hasComponent<TransformComponent>())
    {
        m_beforeScale = m_target->getComponent<TransformComponent>().transform().scale();
    }
}

void SceneObjectScaleHandler::scaleToWorld(const glm::mat4 &worldMatrix)
{
    if (!m_target || !m_target->hasComponent<TransformComponent>())
    {
        return;
    }

    glm::mat4 parentWorld(1.0f);
    if (m_target->parent())
    {
        parentWorld = m_target->scene().getSceneObject(*m_target->parent()).worldMatrix();
    }
    const glm::mat4 localMatrix = glm::inverse(parentWorld) * worldMatrix;

    // Rotation is unchanged by a local scale operation. Project each local basis column onto
    // that rotation axis to retain signed scale and discard small matrix-manipulation drift.
    TransformComponent &transform = m_target->getComponent<TransformComponent>();
    const glm::mat3      rotation  = glm::mat3_cast(transform.transform().rotation());
    const glm::vec3 scale(glm::dot(glm::vec3(localMatrix[0]), rotation[0]),
                          glm::dot(glm::vec3(localMatrix[1]), rotation[1]),
                          glm::dot(glm::vec3(localMatrix[2]), rotation[2]));
    transform.setScale(scale);
}

void SceneObjectScaleHandler::endDrag()
{
    if (!m_target || !m_target->hasComponent<TransformComponent>())
    {
        return;
    }

    const glm::vec3 afterScale = m_target->getComponent<TransformComponent>().transform().scale();
    if (glm::all(glm::lessThanEqual(glm::abs(m_beforeScale - afterScale), glm::vec3(1e-6f))))
    {
        return;
    }

    m_commandManager.appendCommandWithoutExecuting(
        std::make_unique<ScaleSceneObjectCommand>(*m_target, m_beforeScale, afterScale));
    if (m_commitCallback) m_commitCallback(*m_target);
}

} // namespace lr
