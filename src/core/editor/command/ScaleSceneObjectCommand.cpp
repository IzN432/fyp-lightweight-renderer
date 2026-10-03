#include "ScaleSceneObjectCommand.hpp"

#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"

namespace lr
{

void ScaleSceneObjectCommand::execute()
{
    m_object.getComponent<TransformComponent>().setScale(m_after);
}

void ScaleSceneObjectCommand::undo()
{
    m_object.getComponent<TransformComponent>().setScale(m_before);
}

} // namespace lr
