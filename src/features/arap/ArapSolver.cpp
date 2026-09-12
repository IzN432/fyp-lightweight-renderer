#include "ArapSolver.hpp"

#include <set>

namespace lr
{

bool ArapSolver::precompute(const Mesh &mesh, const std::vector<uint32_t> &anchorIndices,
                            const std::vector<uint32_t> &handleIndices)
{
    m_precomputed = false;

    const auto        &positions   = mesh.positions();
    const Eigen::Index vertexCount = static_cast<Eigen::Index>(positions.size());

    Eigen::MatrixXd V(vertexCount, 3);
    for (Eigen::Index i = 0; i < vertexCount; ++i)
    {
        V.row(i) << positions[i].x, positions[i].y, positions[i].z;
    }

    const Eigen::Index faceCount = static_cast<Eigen::Index>(mesh.faces().size());
    Eigen::MatrixXi    F(faceCount, 3);
    for (Eigen::Index f = 0; f < faceCount; ++f)
    {
        const glm::uvec3 &face = mesh.faces()[f];
        F(f, 0)                = static_cast<int>(mesh.positionIndices()[face.x]);
        F(f, 1)                = static_cast<int>(mesh.positionIndices()[face.y]);
        F(f, 2)                = static_cast<int>(mesh.positionIndices()[face.z]);
    }

    // std::set both dedupes and sorts — guards against an overlapping anchor/handle set as well as
    // giving arap_precomputation a stable, sorted boundary-index list.
    std::set<uint32_t> constrained(anchorIndices.begin(), anchorIndices.end());
    constrained.insert(handleIndices.begin(), handleIndices.end());
    if (constrained.empty())
    {
        return false;
    }

    Eigen::VectorXi b(constrained.size());
    Eigen::Index    i = 0;
    for (uint32_t idx : constrained)
    {
        b(i++) = static_cast<int>(idx);
    }

    const bool ok = igl::arap_precomputation(V, F, 3, b, m_data);
    if (!ok)
    {
        return false;
    }

    m_restPositions = std::move(V);
    m_b             = std::move(b);
    m_precomputed   = true;
    return true;
}

std::vector<glm::vec3> ArapSolver::solve(const std::unordered_map<uint32_t, glm::vec3> &handleTargets,
                                         const std::vector<glm::vec3> &warmStart, int iterations)
{
    if (!m_precomputed)
    {
        return warmStart;
    }

    Eigen::MatrixXd bc(m_b.size(), 3);
    for (Eigen::Index i = 0; i < m_b.size(); ++i)
    {
        const uint32_t idx = static_cast<uint32_t>(m_b(i));
        auto           it  = handleTargets.find(idx);
        if (it != handleTargets.end())
        {
            bc.row(i) << it->second.x, it->second.y, it->second.z;
        } else
        {
            bc.row(i) = m_restPositions.row(idx);
        }
    }

    const Eigen::Index vertexCount = static_cast<Eigen::Index>(warmStart.size());
    Eigen::MatrixXd    U(vertexCount, 3);
    for (Eigen::Index i = 0; i < vertexCount; ++i)
    {
        U.row(i) << warmStart[i].x, warmStart[i].y, warmStart[i].z;
    }

    m_data.max_iter = iterations;
    igl::arap_solve(bc, m_data, U);

    std::vector<glm::vec3> result(warmStart.size());
    for (Eigen::Index i = 0; i < vertexCount; ++i)
    {
        result[i] = glm::vec3(static_cast<float>(U(i, 0)), static_cast<float>(U(i, 1)), static_cast<float>(U(i, 2)));
    }
    return result;
}

} // namespace lr
