#pragma once

#include "ArapSolver.hpp"

#include "core/editor/VertexDragHandler.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/editor/selection/SelectionManager.hpp"
#include "core/editor/command/CommandManager.hpp"

#include <vector>
#include <unordered_set>
#include <glm/vec3.hpp>

namespace lr
{

// Drives the current handle-role vertex set through an ArapSolver. handleRole is an opaque id
// handed to it by ArapPlugin — meaningless to anyone else, just forwarded to
// SelectionManager::getIndicesWithRole.
class ArapDragHandler : public VertexDragHandler
{
public:
    ArapDragHandler(ArapSolver &solver, VertexManager &vertexManager, SelectionManager &selectionManager,
                     VertexRoleId handleRole, CommandManager &commandManager,
                     size_t liveSolveVertexThreshold = 5000)
        : m_solver(solver), m_vertexManager(vertexManager), m_selectionManager(selectionManager),
          m_handleRole(handleRole), m_commandManager(commandManager),
          m_liveSolveVertexThreshold(liveSolveVertexThreshold) {}

    const std::unordered_set<uint32_t> &indices() const override;
    void beginDrag() override;
    void translate(const glm::vec3 &frameDelta) override;
    void endDrag(const glm::vec3 &totalDelta) override;

private:
    ArapSolver &m_solver;
    VertexManager &m_vertexManager;
    SelectionManager &m_selectionManager;
    VertexRoleId m_handleRole;
    CommandManager &m_commandManager;
    size_t m_liveSolveVertexThreshold;

    std::vector<glm::vec3> m_beforeDrag;
    // Backing storage for indices() — SelectionManager::getIndicesWithRole returns by value, but
    // the base class interface returns by const ref (matches SelectionManager's own selected/
    // highlighted accessors), so this is recomputed and cached on each call.
    mutable std::unordered_set<uint32_t> m_indicesCache;
};

} // namespace lr
