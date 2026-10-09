#include "EditComponentValuesCommand.hpp"

#include "core/scene/Component.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"

namespace lr
{

void EditComponentValuesCommand::restore(const ComponentValues &values)
{
    if (!m_scene.contains(m_object))
    {
        return;
    }
    if (Component *component = m_scene.getSceneObject(m_object).componentOfType(m_componentType))
    {
        component->restoreUndoValues(values);
    }
}

void EditComponentValuesCommand::execute() { restore(*m_after); }

void EditComponentValuesCommand::undo() { restore(*m_before); }

} // namespace lr
