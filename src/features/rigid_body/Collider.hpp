#pragma once

#include "features/rigid_body/PhysicsMaterial.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <variant>

namespace lr
{

struct SphereCollider
{
    float radius = 0.5f;
};

// A finite rectangle on the local-space plane dot((0,1,0), position) = offset; orient it via the
// owning Collider's localRotation. halfExtents span the local X/Z tangent axes of that plane.
struct PlaneCollider
{
    float     offset = 0.0f;
    glm::vec2 halfExtents{5.0f, 5.0f};
};

struct BoxCollider
{
    glm::vec3 halfExtents{0.5f};
};

using ColliderShape = std::variant<SphereCollider, PlaneCollider, BoxCollider>;

struct Collider
{
    ColliderShape shape{SphereCollider{}};

    // Shape pose relative to the owning SceneObject's transform.
    glm::vec3 localPosition{0.0f};
    glm::quat localRotation{1.0f, 0.0f, 0.0f, 0.0f};

    PhysicsMaterial material;
};

} // namespace lr
