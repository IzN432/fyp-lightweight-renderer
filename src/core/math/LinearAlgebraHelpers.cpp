#include "LinearAlgebraHelpers.hpp"

#include <glm/geometric.hpp>

namespace lr::math
{

glm::vec3 closestPointOnLineToLine(const glm::vec3 &origin, const glm::vec3 &dir, const glm::vec3 &targetOrigin, const glm::vec3 &targetDir)
{
    const glm::vec3 r = origin - targetOrigin;
    const float a     = glm::dot(dir, dir);
    const float b     = glm::dot(dir, targetDir);
    const float c     = glm::dot(targetDir, targetDir);
    const float d     = glm::dot(dir, r);
    const float e     = glm::dot(targetDir, r);

    const float denom = a * c - b * b;
    if (denom < 1e-8f)
    {
        // Lines are parallel: any point on the line is equally close, so fall back to the origin.
        return origin;
    }

    const float t = (b * e - c * d) / denom;
    return origin + t * dir;
}

glm::vec4 planeFromNormalAndPoint(const glm::vec3 &normal, const glm::vec3 &point)
{
    float d = -glm::dot(normal, point);
    return glm::vec4(normal, d);
}

glm::vec3 intersectionBetweenRayAndPlane(const glm::vec3 &rayOrigin, const glm::vec3 &rayDir, const glm::vec4 &plane)
{
    float denom = glm::dot(glm::vec3(plane), rayDir);
    if (std::abs(denom) < 1e-6f)
    {
        // Ray is parallel to the plane: return a point at infinity.
        return glm::vec3(std::numeric_limits<float>::infinity());
    }

    float t = -(glm::dot(glm::vec3(plane), rayOrigin) + plane.w) / denom;
    return rayOrigin + t * rayDir;

}

} // namespace lr::math
