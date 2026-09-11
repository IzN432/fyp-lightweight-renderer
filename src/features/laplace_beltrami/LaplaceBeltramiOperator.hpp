#pragma once

#include <glm/vec3.hpp>

#include <span>
#include <vector>

namespace lr
{

// Pure geometry operation. Triangle indices address positions directly; render vertices, mesh
// attributes, color mapping, and GPU resources are deliberately outside this interface.
class LaplaceBeltramiOperator
{
public:
    static std::vector<float> calculateMagnitude(std::span<const glm::vec3> positions,
                                                 std::span<const glm::uvec3> triangles);
};

} // namespace lr
