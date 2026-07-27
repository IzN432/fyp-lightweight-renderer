#pragma once

#include <glm/glm.hpp>

#include <cstdint>

namespace lr
{

enum class OverlayPrimitive : uint32_t
{
    Cube   = 0,
    Sphere = 1,
    Arrow  = 2,
};

struct OverlayInstance
{
    OverlayPrimitive primitive       = OverlayPrimitive::Cube;
    glm::vec3        position        = {0.0f, 0.0f, 0.0f};
    glm::vec3        eulerDegrees    = {0.0f, 0.0f, 0.0f}; // GLM YXZ intrinsic order
    glm::vec3        scale           = {1.0f, 1.0f, 1.0f};
    glm::vec3        color           = {1.0f, 0.0f, 1.0f};
    float            occludedOpacity = 0.0f;
    uint32_t         pickingId       = ~0u; // ~0u = no picking, 1 = first instance, etc.
};

} // namespace lr
