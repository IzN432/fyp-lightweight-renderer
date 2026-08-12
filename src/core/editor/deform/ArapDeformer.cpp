#include "ArapDeformer.hpp"

#include <igl/ARAPEnergyType.h>
#include <igl/arap.h>

#include <Eigen/Core>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <unordered_set>

namespace lr
{

struct ArapDeformer::Impl
{
    igl::ARAPData  data;
    Eigen::MatrixXd V;  // rest positions, n x 3 — kept for reference/debugging only
    Eigen::MatrixXi F;  // faces, m x 3
    Eigen::VectorXi b;  // constrained vertex indices, matches ArapDeformer::m_constrainedIndices
    Eigen::MatrixXd U;  // current solution, n x 3 — warm start for the next solve()
};

ArapDeformer::ArapDeformer()
    : m_impl(std::make_unique<Impl>())
{
}

ArapDeformer::~ArapDeformer() = default;

bool ArapDeformer::precompute(const std::vector<glm::vec3> &restPositions,
                               const std::vector<glm::uvec3> &faces,
                               const std::vector<uint32_t> &constrainedIndices)
{
    m_precomputed = false;

    Eigen::MatrixXd V(static_cast<Eigen::Index>(restPositions.size()), 3);
    for (size_t i = 0; i < restPositions.size(); ++i)
        V.row(static_cast<Eigen::Index>(i)) =
            Eigen::RowVector3d(restPositions[i].x, restPositions[i].y, restPositions[i].z);

    Eigen::MatrixXi F(static_cast<Eigen::Index>(faces.size()), 3);
    for (size_t i = 0; i < faces.size(); ++i)
        F.row(static_cast<Eigen::Index>(i)) = Eigen::RowVector3i(
            static_cast<int>(faces[i].x), static_cast<int>(faces[i].y), static_cast<int>(faces[i].z));

    m_constrainedIndices = constrainedIndices;
    std::sort(m_constrainedIndices.begin(), m_constrainedIndices.end());
    m_constrainedIndices.erase(std::unique(m_constrainedIndices.begin(), m_constrainedIndices.end()),
                                m_constrainedIndices.end());

    if (m_constrainedIndices.empty())
        return false;

    Eigen::VectorXi b(static_cast<Eigen::Index>(m_constrainedIndices.size()));
    for (size_t i = 0; i < m_constrainedIndices.size(); ++i)
        b(static_cast<Eigen::Index>(i)) = static_cast<int>(m_constrainedIndices[i]);

    // libigl's arap_precomputation hardcodes a Cholesky (positive-definite) factorization of the
    // cotangent-Laplacian system restricted to the *free* (unconstrained) vertices. Any free vertex
    // that isn't referenced by a single face contributes an all-zero row/column to that system,
    // making it singular — Cholesky then fails and arap_precomputation returns false with no other
    // indication why. Check for that specific, very common case up front so the failure is
    // diagnosable instead of a bare "precomputation failed".
    {
        std::vector<bool> referenced(restPositions.size(), false);
        for (const auto &f : faces)
        {
            if (f.x < referenced.size()) referenced[f.x] = true;
            if (f.y < referenced.size()) referenced[f.y] = true;
            if (f.z < referenced.size()) referenced[f.z] = true;
        }

        const std::unordered_set<uint32_t> constrainedLookup(m_constrainedIndices.begin(), m_constrainedIndices.end());

        size_t orphanFreeCount = 0;
        for (size_t i = 0; i < referenced.size(); ++i)
            if (!referenced[i] && !constrainedLookup.count(static_cast<uint32_t>(i)))
                ++orphanFreeCount;

        if (orphanFreeCount > 0)
            spdlog::warn("ArapDeformer::precompute: {} free (unconstrained) vertex/vertices are not "
                         "referenced by any face — this makes the ARAP system singular and Cholesky "
                         "factorization WILL fail. Either include these vertices as anchors/handles, "
                         "or clean the mesh of unreferenced vertices before using ARAP.",
                         orphanFreeCount);
    }

    // igl::ARAPData holds a non-copy-assignable sparse factorization, so a fresh Impl (and thus a
    // fresh, default-constructed ARAPData) is allocated per precompute() rather than reusing the
    // old one via assignment.
    m_impl        = std::make_unique<Impl>();
    m_impl->V     = std::move(V);
    m_impl->F     = std::move(F);
    m_impl->b     = std::move(b);
    m_impl->data.max_iter = 5;

    m_precomputed = igl::arap_precomputation(m_impl->V, m_impl->F, 3, m_impl->b, m_impl->data);
    m_impl->U     = m_impl->V;  // warm start from rest pose until the first solve() replaces it

    return m_precomputed;
}

std::vector<glm::vec3> ArapDeformer::solve(const std::vector<glm::vec3> &constraintPositions)
{
    if (!m_precomputed || constraintPositions.size() != m_constrainedIndices.size())
        return {};

    Eigen::MatrixXd bc(static_cast<Eigen::Index>(constraintPositions.size()), 3);
    for (size_t i = 0; i < constraintPositions.size(); ++i)
        bc.row(static_cast<Eigen::Index>(i)) = Eigen::RowVector3d(
            constraintPositions[i].x, constraintPositions[i].y, constraintPositions[i].z);

    igl::arap_solve(bc, m_impl->data, m_impl->U);

    std::vector<glm::vec3> result(static_cast<size_t>(m_impl->U.rows()));
    for (Eigen::Index i = 0; i < m_impl->U.rows(); ++i)
        result[static_cast<size_t>(i)] = glm::vec3(m_impl->U(i, 0), m_impl->U(i, 1), m_impl->U(i, 2));

    return result;
}

}  // namespace lr
