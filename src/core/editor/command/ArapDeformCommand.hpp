#pragma once

#include "Command.hpp"
#include "core/editor/VertexManager.hpp"

#include <glm/vec3.hpp>
#include <vector>

namespace lr
{

// Records a full-mesh position snapshot before/after an ARAP solve, since — unlike a rigid
// TranslatePointsCommand — the solve can move every vertex, not just the dragged handles.
// Constructed after the deformation has already been applied (see ArapController::update), so
// execute() just re-applies the known "after" state rather than recomputing anything.
class ArapDeformCommand : public Command
{
public:
    ArapDeformCommand(VertexManager &vertexManager, std::vector<glm::vec3> before, std::vector<glm::vec3> after)
        : m_vertexManager(vertexManager), m_before(std::move(before)), m_after(std::move(after)) {}

    void execute() override;
    void undo() override;

private:
    VertexManager &m_vertexManager;
    std::vector<glm::vec3> m_before;
    std::vector<glm::vec3> m_after;
};

}  // namespace lr
