#pragma once

#include "Command.hpp"

#include <glm/vec3.hpp>

namespace lr
{

class SceneObject;

class ScaleSceneObjectCommand : public Command
{
public:
    ScaleSceneObjectCommand(SceneObject &object, glm::vec3 before, glm::vec3 after)
        : m_object(object), m_before(before), m_after(after)
    {}

    void execute() override;
    void undo() override;

private:
    SceneObject &m_object;
    glm::vec3    m_before;
    glm::vec3    m_after;
};

} // namespace lr
