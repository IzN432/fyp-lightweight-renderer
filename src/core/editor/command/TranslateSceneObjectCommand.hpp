#pragma once

#include "Command.hpp"

#include <glm/vec3.hpp>

namespace lr
{

class SceneObject;

class TranslateSceneObjectCommand : public Command
{
public:
    TranslateSceneObjectCommand(SceneObject &object, glm::vec3 localTranslation)
        : m_object(object), m_localTranslation(localTranslation)
    {}

    void execute() override;
    void undo() override;

private:
    void translate(const glm::vec3 &delta);

    SceneObject &m_object;
    glm::vec3    m_localTranslation;
};

} // namespace lr
