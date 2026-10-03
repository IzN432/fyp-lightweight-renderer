#include "SceneObjectRotationHandler.hpp"

#include "core/editor/command/RotateSceneObjectCommand.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <glm/gtc/matrix_inverse.hpp>
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
    if (!m_recordCommands || !m_target || !m_target->hasComponent<TransformComponent>())
    {
        return;
    }

    glm::mat4 parentWorld(1.0f);
    if (m_target->parent())
    {
        parentWorld = m_target->scene().getSceneObject(*m_target->parent()).worldMatrix();
    }
    const glm::mat4 localMatrix = glm::inverse(parentWorld) * worldMatrix;

    // ImGuizmo preserves the object's scale. Divide it back out explicitly so negative scale is
    // handled without turning the extracted rotation basis into a reflection.
    TransformComponent &transform = m_target->getComponent<TransformComponent>();
    const glm::vec3      scale     = transform.transform().scale();
    constexpr float      epsilon   = 1e-6f;
    if (glm::abs(scale.x) <= epsilon || glm::abs(scale.y) <= epsilon || glm::abs(scale.z) <= epsilon)
    {
        return;
    }

    glm::mat3 rotationMatrix;
    rotationMatrix[0] = glm::vec3(localMatrix[0]) / scale.x;
    rotationMatrix[1] = glm::vec3(localMatrix[1]) / scale.y;
    rotationMatrix[2] = glm::vec3(localMatrix[2]) / scale.z;

    // Normalize away small numerical drift accumulated by matrix manipulation.
    rotationMatrix[0] = glm::normalize(rotationMatrix[0]);
    rotationMatrix[1] = glm::normalize(rotationMatrix[1] -
                                       glm::dot(rotationMatrix[1], rotationMatrix[0]) * rotationMatrix[0]);
    rotationMatrix[2] = glm::normalize(glm::cross(rotationMatrix[0], rotationMatrix[1]));
    transform.setRotation(glm::normalize(glm::quat_cast(rotationMatrix)));
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
