#include "features/arap/ArapSolver.hpp"
#include "features/arap/ArapSurfaceFrames.hpp"

#include "core/scene/EngineConventions.hpp"

#include <glm/geometric.hpp>

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
    assert(solver.precompute(mesh, lr::weldedFaces(mesh), {0}, {2}));

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
    assert(solver.precompute(mesh, lr::weldedFaces(mesh), {0, 4}, {2, 6}));

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
    assert(!solver.precompute(mesh, lr::weldedFaces(mesh), {0}, {2, 6}));
    assert(!solver.isPrecomputed());
}

// Lifting the right edge of the first quad out of its plane must tilt its normals and refit its
// tangents, while the inactive second quad keeps exactly the attributes it was given.
void surfaceFramesFollowDeformation()
{
    // A 2x1-quad strip in the z=0 plane (vertices 0-5; middle column 1,4 free) plus an inactive quad
    // (vertices 6-9).
    lr::Mesh mesh;
    mesh.setTopology({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0},
                      {5, 0, 0}, {6, 0, 0}, {6, 1, 0}, {5, 1, 0}},
                     {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                     {{0, 1, 4}, {0, 4, 3}, {1, 2, 5}, {1, 5, 4}, {6, 7, 8}, {6, 8, 9}});
    const glm::vec3 restNormal{0.0f, 0.0f, 1.0f};
    const glm::vec3 inactiveNormal{0.0f, 1.0f, 0.0f}; // deliberately not the geometric normal
    const glm::vec4 inactiveTangent{0.0f, 0.0f, 1.0f, -1.0f};
    std::vector<glm::vec3> normals(10, restNormal);
    std::vector<glm::vec4> tangents(10, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
    std::vector<glm::vec2> uvs(10);
    for (uint32_t vertex = 0; vertex < 10; ++vertex)
    {
        uvs[vertex] = glm::vec2(mesh.positions()[vertex]);
        if (vertex >= 6)
        {
            normals[vertex]  = inactiveNormal;
            tangents[vertex] = inactiveTangent;
        }
    }
    mesh.setPerVertexArray<glm::vec3>(lr::conventions::normalAttribute, normals);
    mesh.setPerVertexArray<glm::vec4>(lr::conventions::tangentAttribute, tangents);
    mesh.setPerVertexArray<glm::vec2>(lr::conventions::uvAttribute, uvs);

    lr::ArapSolver solver;
    assert(solver.precompute(mesh, lr::weldedFaces(mesh), {0, 3}, {2, 5}));
    const auto rest = mesh.positions();
    const glm::vec3 lift(0.0f, 0.0f, 0.5f);
    const auto result = solver.solve({{2, rest[2] + lift}, {5, rest[5] + lift}}, rest, 16);
    mesh.setPositions(result);

    lr::recomputeNormals(mesh, solver.faces(), solver.solvedVertices());
    lr::regenerateTangents(mesh, solver.solvedVertices());

    const auto newNormals  = mesh.getPerVertexArray<glm::vec3>(lr::conventions::normalAttribute);
    const auto newTangents = mesh.getPerVertexArray<glm::vec4>(lr::conventions::tangentAttribute);
    for (uint32_t vertex = 0; vertex < 6; ++vertex)
    {
        const glm::vec3 n = newNormals[vertex];
        assert(std::abs(glm::length(n) - 1.0f) < 1e-3f);
        // Lifting the +x end (a rotation about y) tips the surface normal towards -x.
        assert(n.x < -0.05f && n.z > 0.5f);
        const glm::vec3 t(newTangents[vertex]);
        assert(std::abs(glm::length(t) - 1.0f) < 1e-3f);
        assert(std::abs(glm::dot(t, n)) < 1e-3f);
    }
    for (uint32_t vertex = 6; vertex < 10; ++vertex)
    {
        assert(newNormals[vertex] == inactiveNormal);
        assert(newTangents[vertex] == inactiveTangent);
    }
}

} // namespace

int main()
{
    inactiveComponentRemainsUnchanged();
    everyHandledComponentIsSolved();
    handledComponentWithoutAnchorIsRejected();
    surfaceFramesFollowDeformation();
}
