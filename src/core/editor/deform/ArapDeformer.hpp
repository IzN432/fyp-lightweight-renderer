#pragma once

#include <glm/vec3.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace lr
{

// Wraps libigl's As-Rigid-As-Possible surface deformation (igl::arap_precomputation /
// igl::arap_solve).
//
// precompute() factorizes a sparse system for a given mesh + constrained-vertex set (anchors and
// handles together) and must be re-run whenever either the mesh topology or that set changes.
// solve() reuses the cached factorization to run a few local-global iterations and is cheap
// enough to call off the main thread per interaction (see ArapController).
//
// This class is pure CPU/Eigen state — it never touches Vulkan, VertexManager, or any scene
// object — so it's safe to call solve() from a worker thread as long as no two calls run
// concurrently on the same instance (ArapController enforces that with its busy flag).
class ArapDeformer
{
public:
    ArapDeformer();
    ~ArapDeformer();

    ArapDeformer(const ArapDeformer &)            = delete;
    ArapDeformer &operator=(const ArapDeformer &) = delete;

    // Rebuilds the factorized system for restPositions/faces, constrained to stay fixed (anchors)
    // or be driven externally (handles) at exactly the vertex indices in constrainedIndices.
    // Duplicates are removed and the set is sorted; use constrainedIndices() afterward to learn
    // the order solve()'s constraintPositions must be supplied in.
    // Returns false if libigl's precomputation fails (e.g. an empty or degenerate constraint set).
    bool precompute(const std::vector<glm::vec3> &restPositions,
                     const std::vector<glm::uvec3> &faces,
                     const std::vector<uint32_t> &constrainedIndices);

    // Runs a handful of local-global iterations, warm-started from the previous solve (or the
    // rest pose on the first call after precompute()). constraintPositions must have exactly one
    // entry per index returned by constrainedIndices(), in that same order. Returns the deformed
    // position of every vertex in the mesh (same size/order as restPositions).
    std::vector<glm::vec3> solve(const std::vector<glm::vec3> &constraintPositions);

    bool isPrecomputed() const { return m_precomputed; }

    // The sorted, de-duplicated constrained-vertex indices from the last successful precompute().
    const std::vector<uint32_t> &constrainedIndices() const { return m_constrainedIndices; }

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    bool                   m_precomputed = false;
    std::vector<uint32_t>  m_constrainedIndices;
};

}  // namespace lr
