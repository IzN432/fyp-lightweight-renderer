#include "RenameSceneObjectCommand.hpp"

#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"

namespace lr
{

void RenameSceneObjectCommand::setName(const std::string &name)
{
    if (!m_scene.contains(m_object))
    {
        return;
    }
    m_scene.getSceneObject(m_object).name = name;
}

void RenameSceneObjectCommand::execute() { setName(m_after); }

void RenameSceneObjectCommand::undo() { setName(m_before); }

} // namespace lr
