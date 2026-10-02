#include "TranslateSceneObjectCommand.hpp"

#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"

namespace lr
{

void TranslateSceneObjectCommand::translate(const glm::vec3 &delta)
{
    TransformComponent &transform = m_object.getComponent<TransformComponent>();
    transform.setPosition(transform.transform().position() + delta);
}

void TranslateSceneObjectCommand::execute()
{
    translate(m_localTranslation);
}

void TranslateSceneObjectCommand::undo()
{
    translate(-m_localTranslation);
}

} // namespace lr
