#pragma once

#include "ScaleDragHandler.hpp"

#include "core/editor/command/CommandManager.hpp"
#include "core/scene/SceneObject.hpp"

#include <glm/vec3.hpp>

namespace lr
{

// Applies a world-space ImGuizmo result to an object's parent-local scale and records the
// completed interaction as one undoable command.
class SceneObjectScaleHandler : public ScaleDragHandler
{
public:
    explicit SceneObjectScaleHandler(CommandManager &commandManager) : m_commandManager(commandManager) {}

    void setTarget(SceneObject *target) { m_target = target; }
    SceneObject *target() const { return m_target; }
    void setRecordCommands(bool record) { m_recordCommands = record; }

    void beginDrag() override;
    void scaleToWorld(const glm::mat4 &worldMatrix) override;
    void endDrag() override;

private:
    CommandManager &m_commandManager;
    SceneObject    *m_target = nullptr;
    glm::vec3       m_beforeScale{1.0f};
    bool            m_recordCommands = true;
};

} // namespace lr
