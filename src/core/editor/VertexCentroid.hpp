#pragma once

#include <cstdint>
#include <glm/vec3.hpp>
#include <unordered_set>

namespace lr
{

struct TransformComponent;
class VertexManager;

// World-space centroid of a vertex index set — where a translate gizmo driving those vertices
// belongs. Shared because every feature that drags a vertex subset needs the same answer.
// An empty index set yields the object's own world origin.
glm::vec3 worldVertexCentroid(const VertexManager &vertices, const TransformComponent &transform,
                              const std::unordered_set<uint32_t> &indices);

} // namespace lr
