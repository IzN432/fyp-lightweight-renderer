#include "RotateSceneObjectCommand.hpp"

#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"

namespace lr
{

void RotateSceneObjectCommand::execute()
{
    m_object.getComponent<TransformComponent>().setRotation(m_after);
}

void RotateSceneObjectCommand::undo()
{
    m_object.getComponent<TransformComponent>().setRotation(m_before);
}

} // namespace lr
