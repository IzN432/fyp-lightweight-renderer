#pragma once

#include "core/scene/Mesh.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <glm/vec3.hpp>

#include <Eigen/Core>

namespace lr
{

struct ArapPerformanceStats
{
    double precomputeMs       = 0.0; // Entire precompute(), including mesh/Eigen conversion.
    double solverPrecomputeMs = 0.0; // Selected backend's precompute() only.
    double lastSolveMs        = 0.0; // Selected backend's solve() only.
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

class ArapBackend;

// Converts engine Mesh/GLM data to Eigen and delegates the numerical work to the selected backend.
// Pure numerics — no dependency on SelectionManager, VertexManager, gizmos, or undo.
class ArapSolver
{
public:
    ArapSolver();
    ~ArapSolver();

    ArapSolver(const ArapSolver &)            = delete;
    ArapSolver &operator=(const ArapSolver &) = delete;

    // anchorIndices/handleIndices index into mesh.positions() (the deduped vertex space). Only
    // components containing handles are solved; other disconnected components stay unchanged.
    // Returns false if an active component has no anchor or backend precomputation fails.
    bool precompute(const Mesh &mesh, const std::vector<uint32_t> &anchorIndices,
                    const std::vector<uint32_t> &handleIndices);

    bool isPrecomputed() const { return m_precomputed; }
    std::string_view backendName() const;
    void invalidate() { m_precomputed = false; }
    const ArapPerformanceStats &performanceStats() const { return m_stats; }
    void recordInteraction(double elapsedMs, bool release);

    // handleTargets: absolute target position for every handle vertex this call (b-indices not
    // present are assumed to be anchors, held fixed at their precompute-time rest position).
    // initialPositions: current position for every vertex in the mesh (size == mesh vertex count) —
    // the iterative solve starts from these values. iterations is forwarded to the selected backend
    // for this call only. Returns the solved position for every vertex (size ==
    // initialPositions.size()). No-op (returns initialPositions unchanged) if not precomputed.
    std::vector<glm::vec3> solve(const std::unordered_map<uint32_t, glm::vec3> &handleTargets,
                                 const std::vector<glm::vec3> &initialPositions, int iterations);

private:
    struct Component
    {
        std::unique_ptr<ArapBackend> backend;
        std::vector<uint32_t>        globalIndices;
        Eigen::MatrixXd              restPositions;
        Eigen::VectorXi              constrainedLocalIndices;
    };

    std::vector<Component> m_components;
    std::string            m_backendName;
    bool                 m_precomputed = false;
    ArapPerformanceStats m_stats;
    double               m_totalSolveMs = 0.0;
};

} // namespace lr
