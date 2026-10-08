#include "features/arap/ArapSolver.hpp"

#include <cassert>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace
{

lr::Mesh makeDisconnectedQuads()
{
    lr::Mesh mesh;
    mesh.setTopology({{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
                      {3.0f, 0.0f, 0.0f}, {4.0f, 0.0f, 0.0f}, {4.0f, 1.0f, 0.0f}, {3.0f, 1.0f, 0.0f}},
                     {0, 1, 2, 3, 4, 5, 6, 7}, {{0, 1, 2}, {0, 2, 3}, {4, 5, 6}, {4, 6, 7}});
    return mesh;
}

bool near(const glm::vec3 &a, const glm::vec3 &b)
{
    return std::abs(a.x - b.x) < 1e-4f && std::abs(a.y - b.y) < 1e-4f && std::abs(a.z - b.z) < 1e-4f;
}

void inactiveComponentRemainsUnchanged()
{
    const lr::Mesh mesh = makeDisconnectedQuads();
    lr::ArapSolver solver;
    assert(solver.precompute(mesh, {0}, {2}));

    const auto rest = mesh.positions();
    const glm::vec3 target = rest[2] + glm::vec3(0.5f, 0.0f, 0.0f);
    const auto result = solver.solve({{2, target}}, rest, 8);

    assert(near(result[0], rest[0]));
    assert(near(result[2], target));
    for (uint32_t vertex = 4; vertex < 8; ++vertex)
    {
        assert(result[vertex] == rest[vertex]);
    }
}

void everyHandledComponentIsSolved()
{
    const lr::Mesh mesh = makeDisconnectedQuads();
    lr::ArapSolver solver;
    assert(solver.precompute(mesh, {0, 4}, {2, 6}));

    const auto rest = mesh.positions();
    const glm::vec3 firstTarget  = rest[2] + glm::vec3(0.5f, 0.0f, 0.0f);
    const glm::vec3 secondTarget = rest[6] + glm::vec3(-0.5f, 0.0f, 0.0f);
    const auto result = solver.solve({{2, firstTarget}, {6, secondTarget}}, rest, 8);

    assert(near(result[0], rest[0]));
    assert(near(result[4], rest[4]));
    assert(near(result[2], firstTarget));
    assert(near(result[6], secondTarget));
}

void handledComponentWithoutAnchorIsRejected()
{
    const lr::Mesh mesh = makeDisconnectedQuads();
    lr::ArapSolver solver;
    assert(!solver.precompute(mesh, {0}, {2, 6}));
    assert(!solver.isPrecomputed());
}

} // namespace

int main()
{
    inactiveComponentRemainsUnchanged();
    everyHandledComponentIsSolved();
    handledComponentWithoutAnchorIsRejected();
}
