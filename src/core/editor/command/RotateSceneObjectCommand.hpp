#pragma once

#include "Command.hpp"

#include <glm/gtc/quaternion.hpp>

namespace lr
{

class SceneObject;

class RotateSceneObjectCommand : public Command
{
public:
    RotateSceneObjectCommand(SceneObject &object, glm::quat before, glm::quat after)
        : m_object(object), m_before(before), m_after(after)
    {}

    void execute() override;
    void undo() override;

private:
    SceneObject &m_object;
    glm::quat    m_before;
    glm::quat    m_after;
};

} // namespace lr
