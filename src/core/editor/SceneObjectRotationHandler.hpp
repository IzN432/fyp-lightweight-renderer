#pragma once

#include "RotateDragHandler.hpp"

#include "core/editor/command/CommandManager.hpp"
#include "core/scene/SceneObject.hpp"

#include <glm/gtc/quaternion.hpp>

namespace lr
{

// Applies a world-space ImGuizmo result to an object's parent-local rotation and records the
// completed interaction as one undoable command.
class SceneObjectRotationHandler : public RotateDragHandler
{
public:
    explicit SceneObjectRotationHandler(CommandManager &commandManager) : m_commandManager(commandManager) {}

    void setTarget(SceneObject *target) { m_target = target; }
    SceneObject *target() const { return m_target; }
    void setRecordCommands(bool record) { m_recordCommands = record; }

    void beginDrag() override;
    void rotateToWorld(const glm::mat4 &worldMatrix) override;
    void endDrag() override;

private:
    CommandManager &m_commandManager;
    SceneObject    *m_target = nullptr;
    glm::quat       m_beforeRotation{1.0f, 0.0f, 0.0f, 0.0f};
    bool            m_recordCommands = true;
};

} // namespace lr
