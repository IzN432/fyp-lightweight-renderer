#pragma once

#include "core/scene/Mesh.hpp"

#include <Eigen/Core>

#include <vector>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace lr
{

// The per-render-vertex shading frame ARAP has to keep in step with the positions it moves.
// An empty array means the mesh has no such attribute.
struct SurfaceFrames
{
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> tangents;
};

SurfaceFrames captureSurfaceFrames(const Mesh &mesh);
// Writes back whichever arrays are non-empty (one revision each).
void restoreSurfaceFrames(Mesh &mesh, const SurfaceFrames &frames);

// Rebuilds "normal" from the current positions as angle-weighted averages of the welded faces'
// normals (igl::per_vertex_normals), so split render vertices sharing a position get the same
// smooth normal. Only render vertices whose position is flagged in solvedVertices are written.
// No-op when the mesh has no normal attribute.
void recomputeNormals(Mesh &mesh, const Eigen::MatrixXi &weldedFaces, const std::vector<bool> &solvedVertices);

// Regenerates "tangent" with MikkTSpace from the current positions, normals and uvs, writing only
// render vertices whose position is flagged in solvedVertices. Too slow to run every drag frame,
// so it runs once on release. No-op when the mesh lacks a normal, tangent or uv attribute.
void regenerateTangents(Mesh &mesh, const std::vector<bool> &solvedVertices);

} // namespace lr
