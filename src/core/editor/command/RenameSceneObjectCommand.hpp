#pragma once

#include "Command.hpp"

#include "core/scene/SceneObjectId.hpp"

#include <string>

namespace lr
{

class Scene;

// A rename from the Scene Hierarchy's context menu.
class RenameSceneObjectCommand : public Command
{
public:
    RenameSceneObjectCommand(Scene &scene, SceneObjectId object, std::string before, std::string after)
        : m_scene(scene), m_object(object), m_before(std::move(before)), m_after(std::move(after))
    {}

    void execute() override;
    void undo() override;

private:
    void setName(const std::string &name);

    Scene        &m_scene;
    SceneObjectId m_object;
    std::string   m_before;
    std::string   m_after;
};

} // namespace lr
