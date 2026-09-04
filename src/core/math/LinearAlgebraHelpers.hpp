#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.h>

namespace lr::math
{

// Returns the point on the line (origin, dir) that is closest to the line (targetOrigin, targetDir).
glm::vec3 closestPointOnLineToLine(const glm::vec3 &origin, const glm::vec3 &dir, const glm::vec3 &targetOrigin,
                                   const glm::vec3 &targetDir);
glm::vec4 planeFromNormalAndPoint(const glm::vec3 &normal, const glm::vec3 &point);
glm::vec3 intersectionBetweenRayAndPlane(const glm::vec3 &rayOrigin, const glm::vec3 &rayDir, const glm::vec4 &plane);

// Projects a world-space point to pixel coordinates. Matches the no-Y-flip NDC convention used by
// SelectionManager/GizmoManager's mouse-to-NDC conversion (NDC [-1,1] maps linearly to [0,extent]).
glm::vec2 worldToScreenPixels(const glm::vec3 &worldPos, const glm::mat4 &viewProj, VkExtent2D extent);

} // namespace lr::math
