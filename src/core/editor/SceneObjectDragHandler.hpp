#pragma once

#include "TranslateDragHandler.hpp"

#include "core/editor/command/CommandManager.hpp"
#include "core/scene/SceneObject.hpp"

namespace lr
{

// Adapts the translate gizmos to a SceneObject. Gizmo deltas are expressed in world space;
// objects store their position in parent-local space, so translate() performs that conversion.
class SceneObjectDragHandler : public TranslateDragHandler
{
public:
    explicit SceneObjectDragHandler(CommandManager &commandManager) : m_commandManager(commandManager) {}

    void setTarget(SceneObject *target) { m_target = target; }
    SceneObject *target() const { return m_target; }
    void setRecordCommands(bool record) { m_recordCommands = record; }

    void beginDrag() override;
    void translate(const glm::vec3 &frameDelta) override;
    void endDrag(const glm::vec3 &totalDelta) override;

private:
    glm::vec3 worldDeltaToLocal(const glm::vec3 &worldDelta) const;

    CommandManager                    &m_commandManager;
    SceneObject                       *m_target = nullptr;
    glm::vec3                          m_accumulatedLocalDelta{0.0f};
    bool                               m_recordCommands = true;
};

} // namespace lr
