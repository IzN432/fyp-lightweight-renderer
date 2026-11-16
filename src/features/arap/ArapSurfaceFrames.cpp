#include "ArapSurfaceFrames.hpp"

#include <cstdint>
#include <vector>

#include "core/loaders/LoaderUtils.hpp"
#include "core/scene/EngineConventions.hpp"

#include <igl/per_vertex_normals.h>

#include <cmath>
#include <span>

namespace lr
{

namespace
{

bool hasPerVertex(const Mesh &mesh, const char *name)
{
    return mesh.layout().findPerVertexAttr(name) != nullptr;
}

template <typename T> std::vector<T> copyPerVertex(const Mesh &mesh, const char *name)
{
    const std::span<const T> data = mesh.getPerVertexArray<T>(name);
    return {data.begin(), data.end()};
}

bool isSolved(const std::vector<bool> &solvedVertices, uint32_t position)
{
    return position < solvedVertices.size() && solvedVertices[position];
}

} // namespace

SurfaceFrames captureSurfaceFrames(const Mesh &mesh)
{
    SurfaceFrames frames;
    if (hasPerVertex(mesh, conventions::normalAttribute))
    {
        frames.normals = copyPerVertex<glm::vec3>(mesh, conventions::normalAttribute);
    }
    if (hasPerVertex(mesh, conventions::tangentAttribute))
    {
        frames.tangents = copyPerVertex<glm::vec4>(mesh, conventions::tangentAttribute);
    }
    return frames;
}

void restoreSurfaceFrames(Mesh &mesh, const SurfaceFrames &frames)
{
    if (!frames.normals.empty())
    {
        mesh.setPerVertexArray<glm::vec3>(conventions::normalAttribute, frames.normals);
    }
    if (!frames.tangents.empty())
    {
        mesh.setPerVertexArray<glm::vec4>(conventions::tangentAttribute, frames.tangents);
    }
}

void recomputeNormals(Mesh &mesh, const Eigen::MatrixXi &weldedFaces, const std::vector<bool> &solvedVertices)
{
    if (!hasPerVertex(mesh, conventions::normalAttribute) || weldedFaces.rows() == 0)
    {
        return;
    }

    const auto &positions = mesh.positions();
    Eigen::MatrixXd V(static_cast<Eigen::Index>(positions.size()), 3);
    for (Eigen::Index i = 0; i < V.rows(); ++i)
    {
        V.row(i) << positions[i].x, positions[i].y, positions[i].z;
    }

    Eigen::MatrixXd N;
    igl::per_vertex_normals(V, weldedFaces, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, N);

    std::vector<glm::vec3> normals = copyPerVertex<glm::vec3>(mesh, conventions::normalAttribute);
    const auto            &positionIndices = mesh.positionIndices();
    for (size_t vertex = 0; vertex < normals.size() && vertex < positionIndices.size(); ++vertex)
    {
        const uint32_t position = positionIndices[vertex];
        if (!isSolved(solvedVertices, position))
        {
            continue;
        }
        const glm::vec3 n(static_cast<float>(N(position, 0)), static_cast<float>(N(position, 1)),
                          static_cast<float>(N(position, 2)));
        // per_vertex_normals leaves a zero/NaN row for isolated or fully degenerate vertices; keep
        // the previous normal there rather than upload garbage.
        if (std::isfinite(n.x) && std::isfinite(n.y) && std::isfinite(n.z) && glm::dot(n, n) > 0.0f)
        {
            normals[vertex] = n;
        }
    }
    mesh.setPerVertexArray<glm::vec3>(conventions::normalAttribute, normals);
}

void regenerateTangents(Mesh &mesh, const std::vector<bool> &solvedVertices)
{
    if (!hasPerVertex(mesh, conventions::normalAttribute) || !hasPerVertex(mesh, conventions::tangentAttribute) ||
        !hasPerVertex(mesh, conventions::uvAttribute))
    {
        return;
    }

    MeshData data;
    data.positions       = mesh.positions();
    data.positionIndices = mesh.positionIndices();
    data.faces           = mesh.faces();
    data.normals         = copyPerVertex<glm::vec3>(mesh, conventions::normalAttribute);
    data.uvs             = copyPerVertex<glm::vec2>(mesh, conventions::uvAttribute);
    data.tangents.assign(data.positionIndices.size(), glm::vec4(0.0f));
    generateTangents(data);

    std::vector<glm::vec4> tangents = copyPerVertex<glm::vec4>(mesh, conventions::tangentAttribute);
    for (size_t vertex = 0; vertex < tangents.size() && vertex < data.tangents.size(); ++vertex)
    {
        if (isSolved(solvedVertices, data.positionIndices[vertex]))
        {
            tangents[vertex] = data.tangents[vertex];
        }
    }
    mesh.setPerVertexArray<glm::vec4>(conventions::tangentAttribute, tangents);
}

} // namespace lr
