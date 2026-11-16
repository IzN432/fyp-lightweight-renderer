#include "ArapDragHandler.hpp"

#include "ArapDeformCommand.hpp"

#include <chrono>
#include <memory>
#include <unordered_map>

namespace lr
{

const std::unordered_set<uint32_t> &ArapDragHandler::indices() const
{
    const std::vector<uint32_t> roleIndices = m_selectionManager.getIndicesWithRole(m_handleRole);
    m_indicesCache                          = std::unordered_set<uint32_t>(roleIndices.begin(), roleIndices.end());
    return m_indicesCache;
}

void ArapDragHandler::beginDrag()
{
    m_beforeDrag       = m_vertexManager.getPositions();
    m_framesBeforeDrag = captureSurfaceFrames(m_vertexManager.mesh());
}

void ArapDragHandler::translate(const glm::vec3 &frameDelta)
{
    const auto interactionStart = std::chrono::steady_clock::now();
    const auto &positions     = m_vertexManager.getPositions();
    const auto &handleIndices = indices();
    const glm::vec3 localDelta = worldDeltaToLocal(frameDelta);

    if (positions.size() < m_liveSolveVertexThreshold)
    {
        // Solve every frame — libigl pins constrained rows to bc, so the handles land exactly on
        // target as a side effect; no separate handle-only move is needed.
        std::unordered_map<uint32_t, glm::vec3> targets;
        for (uint32_t idx : handleIndices)
        {
            targets[idx] = positions[idx] + localDelta;
        }

        std::vector<glm::vec3> result = m_solver.solve(targets, positions, /*iterations=*/1);

        std::vector<uint32_t> allIndices(result.size());
        for (uint32_t i = 0; i < allIndices.size(); ++i)
        {
            allIndices[i] = i;
        }
        m_vertexManager.setPositions(allIndices, result);
        // Normals are cheap (one pass over the faces), so they follow the surface live. Tangents
        // wait for endDrag(); geometry.frag re-orthogonalises the stale tangent against the new
        // normal meanwhile, so the TBN basis stays valid.
        if (m_solver.isPrecomputed())
        {
            recomputeNormals(m_vertexManager.mesh(), m_solver.faces(), m_solver.solvedVertices());
        }
    } else
    {
        // Too large to solve live — just move the handles themselves; the rest of the mesh stays
        // put until endDrag(). setPositions needs indices/positions as parallel vectors, so build
        // both from the (unordered) handle set in the same pass.
        std::vector<uint32_t>  handleIndicesVec;
        std::vector<glm::vec3> newHandlePositions;
        handleIndicesVec.reserve(handleIndices.size());
        newHandlePositions.reserve(handleIndices.size());
        for (uint32_t idx : handleIndices)
        {
            handleIndicesVec.push_back(idx);
            newHandlePositions.push_back(positions[idx] + localDelta);
        }
        m_vertexManager.setPositions(handleIndicesVec, newHandlePositions);
    }

    m_solver.recordInteraction(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - interactionStart).count(),
        false);
}

void ArapDragHandler::endDrag(const glm::vec3 &totalDelta)
{
    const auto interactionStart = std::chrono::steady_clock::now();
    const auto &positions     = m_vertexManager.getPositions();
    const auto &handleIndices = indices();

    // Handles are already sitting at their final dragged position (either translate() branch put
    // them there), so the closing solve targets them at their current position — zero additional
    // delta — just with more iterations to fully converge.
    std::unordered_map<uint32_t, glm::vec3> targets;
    for (uint32_t idx : handleIndices)
    {
        targets[idx] = positions[idx];
    }

    std::vector<glm::vec3> result = m_solver.solve(targets, positions, /*iterations=*/12);

    std::vector<uint32_t> allIndices(result.size());
    for (uint32_t i = 0; i < allIndices.size(); ++i)
    {
        allIndices[i] = i;
    }
    m_vertexManager.setPositions(allIndices, result);
    if (m_solver.isPrecomputed())
    {
        Mesh &mesh = m_vertexManager.mesh();
        recomputeNormals(mesh, m_solver.faces(), m_solver.solvedVertices());
        regenerateTangents(mesh, m_solver.solvedVertices());
    }

    // `positions` is a live reference into VertexManager's internal array, so it already reflects
    // the setPositions() call above — safe to diff against it directly.
    std::vector<ArapDeformCommand::VertexDiff> diffs;
    for (size_t i = 0; i < m_beforeDrag.size() && i < positions.size(); ++i)
    {
        if (m_beforeDrag[i] != positions[i])
        {
            diffs.push_back({static_cast<uint32_t>(i), m_beforeDrag[i], positions[i]});
        }
    }

    if (!diffs.empty())
    {
        m_commandManager.appendCommandWithoutExecuting(
            std::make_unique<ArapDeformCommand>(m_vertexManager, std::move(diffs), std::move(m_framesBeforeDrag),
                                                captureSurfaceFrames(m_vertexManager.mesh())));
    }

    m_solver.recordInteraction(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - interactionStart).count(),
        true);
}

} // namespace lr
