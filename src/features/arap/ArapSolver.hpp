#pragma once

#include "core/scene/Mesh.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <glm/vec3.hpp>

#include <Eigen/Core>
#include <igl/arap.h>

namespace lr
{

struct ArapPerformanceStats
{
    double precomputeMs       = 0.0; // Entire precompute(), including mesh/Eigen conversion.
    double solverPrecomputeMs = 0.0; // igl::arap_precomputation() only.
    double lastSolveMs        = 0.0; // igl::arap_solve() only.
    double averageSolveMs     = 0.0;
    double minSolveMs         = 0.0;
    double maxSolveMs         = 0.0;
    double lastInteractionMs  = 0.0; // Drag/release handler, including mesh update callbacks.

    uint64_t solveCount      = 0;
    size_t   vertexCount     = 0;
    size_t   triangleCount   = 0;
    size_t   constraintCount = 0;
    int      lastIterations  = 0;
    bool     lastWasRelease  = false;
};

// Thin wrapper around libigl's ARAP precompute/solve. Pure numerics — plain vertex index lists in,
// no dependency on SelectionManager, VertexManager, gizmos, or undo.
class ArapSolver
{
public:
    // anchorIndices/handleIndices index into mesh.positions() (the deduped vertex space). Returns
    // false on failure (e.g. a free-vertex component that can't reach any anchor/handle).
    bool precompute(const Mesh &mesh, const std::vector<uint32_t> &anchorIndices,
                    const std::vector<uint32_t> &handleIndices);

    bool isPrecomputed() const { return m_precomputed; }
    void invalidate() { m_precomputed = false; }
    const ArapPerformanceStats &performanceStats() const { return m_stats; }
    void recordInteraction(double elapsedMs, bool release);

    // handleTargets: absolute target position for every handle vertex this call (b-indices not
    // present are assumed to be anchors, held fixed at their precompute-time rest position).
    // warmStart: current position for every vertex in the mesh (size == mesh vertex count) — seeds
    // the iterative solve. iterations is forwarded to ARAPData::max_iter for this call only.
    // Returns the solved position for every vertex (size == warmStart.size()). No-op (returns
    // warmStart unchanged) if not currently precomputed.
    std::vector<glm::vec3> solve(const std::unordered_map<uint32_t, glm::vec3> &handleTargets,
                                 const std::vector<glm::vec3> &warmStart, int iterations);

private:
    igl::ARAPData   m_data;
    Eigen::MatrixXd m_restPositions; // V at precompute time — anchor bc targets are read from here
    Eigen::VectorXi m_b;             // sorted anchor+handle indices, as passed to arap_precomputation
    bool                 m_precomputed = false;
    ArapPerformanceStats m_stats;
    double               m_totalSolveMs = 0.0;
};

} // namespace lr
