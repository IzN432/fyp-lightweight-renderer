#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace lr::math
{

// Returns the point on the line (origin, dir) that is closest to the line (targetOrigin, targetDir).
glm::vec3 closestPointOnLineToLine(const glm::vec3 &origin, const glm::vec3 &dir, const glm::vec3 &targetOrigin, const glm::vec3 &targetDir);
glm::vec4 planeFromNormalAndPoint(const glm::vec3 &normal, const glm::vec3 &point);
glm::vec3 intersectionBetweenRayAndPlane(const glm::vec3 &rayOrigin, const glm::vec3 &rayDir, const glm::vec4 &plane);

} // namespace lr::math
