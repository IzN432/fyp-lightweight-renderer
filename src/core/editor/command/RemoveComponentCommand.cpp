#include "RemoveComponentCommand.hpp"

#include "core/editor/EditorContext.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"

namespace lr
{

void RemoveComponentCommand::execute()
{
    if (!m_scene.contains(m_object))
    {
        return;
    }
    m_scene.getSceneObject(m_object).removeComponent(m_componentType);
}

void RemoveComponentCommand::undo()
{
    if (!m_scene.contains(m_object))
    {
        return;
    }
    SceneObject &object = m_scene.getSceneObject(m_object);
    if (object.hasComponent(m_componentType))
    {
        return;
    }
    m_adder(object, *m_values);
    m_addService.onComponentAdded(object);
}

} // namespace lr
