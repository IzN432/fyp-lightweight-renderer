#include "AddComponentCommand.hpp"

#include "core/editor/EditorContext.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"

namespace lr
{

void AddComponentCommand::execute()
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
    // The same notification the Inspector sends when it adds one by hand: an object that has just
    // become renderable again has to be put back in front of the GPU side.
    m_addService.onComponentAdded(object);
}

void AddComponentCommand::undo()
{
    if (!m_scene.contains(m_object))
    {
        return;
    }
    m_scene.getSceneObject(m_object).removeComponent(m_componentType);
}

} // namespace lr
