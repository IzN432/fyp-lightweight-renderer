#include "ArapSolver.hpp"
#include "ArapBackend.hpp"

#include <algorithm>
#include <chrono>
#include <queue>
#include <set>
#include <unordered_set>

namespace lr
{

ArapSolver::ArapSolver()
{
    const auto backend = createArapBackend();
    m_backendName      = backend->name();
}

ArapSolver::~ArapSolver() = default;

std::string_view ArapSolver::backendName() const
{
    return m_backendName;
}

bool ArapSolver::precompute(const Mesh &mesh, const std::vector<uint32_t> &anchorIndices,
                            const std::vector<uint32_t> &handleIndices)
{
    const auto precomputeStart = std::chrono::steady_clock::now();
    m_precomputed  = false;
    m_components.clear();
    m_stats        = {};
    m_totalSolveMs = 0.0;

    const auto        &positions   = mesh.positions();
    const Eigen::Index vertexCount = static_cast<Eigen::Index>(positions.size());
    m_stats.vertexCount            = positions.size();
    m_stats.triangleCount          = mesh.faces().size();

    if (vertexCount == 0)
    {
        return false;
    }

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
    m_stats.constraintCount = constrained.size();
    if (constrained.empty())
    {
        m_stats.precomputeMs = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - precomputeStart)
                                   .count();
        return false;
    }

    for (uint32_t index : constrained)
    {
        if (index >= positions.size())
        {
            return false;
        }
    }

    // Partition the unique-position graph. Only components containing handles participate in the
    // solve; disconnected decorative islands therefore remain unchanged. Every active component
    // must contain an anchor so its rigid motion is grounded independently.
    std::vector<std::vector<uint32_t>> adjacency(positions.size());
    for (Eigen::Index f = 0; f < faceCount; ++f)
    {
        const uint32_t a = static_cast<uint32_t>(F(f, 0));
        const uint32_t b = static_cast<uint32_t>(F(f, 1));
        const uint32_t c = static_cast<uint32_t>(F(f, 2));
        if (a >= positions.size() || b >= positions.size() || c >= positions.size())
        {
            return false;
        }
        adjacency[a].push_back(b);
        adjacency[a].push_back(c);
        adjacency[b].push_back(a);
        adjacency[b].push_back(c);
        adjacency[c].push_back(a);
        adjacency[c].push_back(b);
    }

    const std::unordered_set<uint32_t> anchors(anchorIndices.begin(), anchorIndices.end());
    const std::unordered_set<uint32_t> handles(handleIndices.begin(), handleIndices.end());
    std::vector<int> componentOf(positions.size(), -1);
    std::vector<std::vector<uint32_t>> componentVertices;
    for (uint32_t seed = 0; seed < positions.size(); ++seed)
    {
        if (componentOf[seed] != -1)
        {
            continue;
        }
        const int component = static_cast<int>(componentVertices.size());
        componentVertices.emplace_back();
        std::queue<uint32_t> pending;
        pending.push(seed);
        componentOf[seed] = component;
        while (!pending.empty())
        {
            const uint32_t vertex = pending.front();
            pending.pop();
            componentVertices.back().push_back(vertex);
            for (uint32_t neighbour : adjacency[vertex])
            {
                if (componentOf[neighbour] == -1)
                {
                    componentOf[neighbour] = component;
                    pending.push(neighbour);
                }
            }
        }
    }

    const auto solverStart = std::chrono::steady_clock::now();
    for (size_t componentIndex = 0; componentIndex < componentVertices.size(); ++componentIndex)
    {
        const auto &vertices = componentVertices[componentIndex];
        const bool hasHandle = std::ranges::any_of(vertices, [&](uint32_t v) { return handles.contains(v); });
        if (!hasHandle)
        {
            continue;
        }
        const bool hasAnchor = std::ranges::any_of(vertices, [&](uint32_t v) { return anchors.contains(v); });
        if (!hasAnchor)
        {
            m_stats.solverPrecomputeMs = std::chrono::duration<double, std::milli>(
                                             std::chrono::steady_clock::now() - solverStart)
                                             .count();
            m_stats.precomputeMs = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - precomputeStart)
                                       .count();
            m_components.clear();
            return false;
        }

        Component problem;
        problem.globalIndices = vertices;
        problem.restPositions.resize(static_cast<Eigen::Index>(vertices.size()), 3);
        std::vector<int> localOf(positions.size(), -1);
        for (size_t local = 0; local < vertices.size(); ++local)
        {
            localOf[vertices[local]] = static_cast<int>(local);
            problem.restPositions.row(static_cast<Eigen::Index>(local)) = V.row(vertices[local]);
        }

        std::vector<Eigen::Vector3i> localFaces;
        for (Eigen::Index f = 0; f < faceCount; ++f)
        {
            if (componentOf[static_cast<uint32_t>(F(f, 0))] == static_cast<int>(componentIndex))
            {
                localFaces.emplace_back(localOf[static_cast<uint32_t>(F(f, 0))],
                                        localOf[static_cast<uint32_t>(F(f, 1))],
                                        localOf[static_cast<uint32_t>(F(f, 2))]);
            }
        }
        Eigen::MatrixXi localF(static_cast<Eigen::Index>(localFaces.size()), 3);
        for (Eigen::Index f = 0; f < localF.rows(); ++f)
        {
            localF.row(f) = localFaces[static_cast<size_t>(f)];
        }

        std::vector<int> localConstraints;
        for (size_t local = 0; local < vertices.size(); ++local)
        {
            if (constrained.contains(vertices[local]))
            {
                localConstraints.push_back(static_cast<int>(local));
            }
        }
        problem.constrainedLocalIndices = Eigen::Map<Eigen::VectorXi>(localConstraints.data(),
                                                                       localConstraints.size());
        problem.backend = createArapBackend();
        if (!problem.backend->precompute(problem.restPositions, localF, problem.constrainedLocalIndices))
        {
            m_components.clear();
            m_stats.solverPrecomputeMs = std::chrono::duration<double, std::milli>(
                                             std::chrono::steady_clock::now() - solverStart)
                                             .count();
            m_stats.precomputeMs = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - precomputeStart)
                                       .count();
            return false;
        }
        m_components.push_back(std::move(problem));
    }
    const bool ok = !m_components.empty();
    m_stats.solverPrecomputeMs = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - solverStart)
                                     .count();
    m_stats.precomputeMs = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - precomputeStart)
                               .count();
    if (!ok)
    {
        return false;
    }

    m_precomputed   = true;
    m_stats.precomputeMs = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - precomputeStart)
                               .count();
    return true;
}

std::vector<glm::vec3> ArapSolver::solve(const std::unordered_map<uint32_t, glm::vec3> &handleTargets,
                                         const std::vector<glm::vec3> &initialPositions, int iterations)
{
    if (!m_precomputed)
    {
        return initialPositions;
    }

    const auto solveStart = std::chrono::steady_clock::now();
    std::vector<glm::vec3> result = initialPositions;
    bool solveOk = true;
    for (Component &problem : m_components)
    {
        Eigen::MatrixXd bc(problem.constrainedLocalIndices.size(), 3);
        for (Eigen::Index i = 0; i < problem.constrainedLocalIndices.size(); ++i)
        {
            const int      local  = problem.constrainedLocalIndices(i);
            const uint32_t global = problem.globalIndices[static_cast<size_t>(local)];
            const auto     target = handleTargets.find(global);
            if (target != handleTargets.end())
            {
                bc.row(i) << target->second.x, target->second.y, target->second.z;
            } else
            {
                bc.row(i) = problem.restPositions.row(local);
            }
        }

        Eigen::MatrixXd U(static_cast<Eigen::Index>(problem.globalIndices.size()), 3);
        for (size_t local = 0; local < problem.globalIndices.size(); ++local)
        {
            const glm::vec3 &position = initialPositions[problem.globalIndices[local]];
            U.row(static_cast<Eigen::Index>(local)) << position.x, position.y, position.z;
        }
        if (!problem.backend->solve(bc, U, iterations))
        {
            solveOk = false;
            break;
        }
        for (size_t local = 0; local < problem.globalIndices.size(); ++local)
        {
            result[problem.globalIndices[local]] =
                glm::vec3(static_cast<float>(U(local, 0)), static_cast<float>(U(local, 1)),
                          static_cast<float>(U(local, 2)));
        }
    }
    const double solveMs = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - solveStart)
                               .count();
    m_stats.lastSolveMs = solveMs;
    m_stats.lastIterations = iterations;
    ++m_stats.solveCount;
    m_totalSolveMs += solveMs;
    m_stats.averageSolveMs = m_totalSolveMs / static_cast<double>(m_stats.solveCount);
    if (m_stats.solveCount == 1)
    {
        m_stats.minSolveMs = solveMs;
        m_stats.maxSolveMs = solveMs;
    } else
    {
        m_stats.minSolveMs = std::min(m_stats.minSolveMs, solveMs);
        m_stats.maxSolveMs = std::max(m_stats.maxSolveMs, solveMs);
    }
    if (!solveOk)
    {
        return initialPositions;
    }

    return result;
}

void ArapSolver::recordInteraction(double elapsedMs, bool release)
{
    m_stats.lastInteractionMs = elapsedMs;
    m_stats.lastWasRelease    = release;
}

} // namespace lr
