#pragma once

#include "features/rigid_body/PhysicsMaterial.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <type_traits>
#include <variant>

namespace lr
{

enum class ColliderShapeType
{
    Sphere,
    Plane,
    Box,
};

struct SphereCollider
{
    static constexpr ColliderShapeType type = ColliderShapeType::Sphere;
    float radius = 0.5f;
};

// A finite rectangle on the local-space plane dot((0,1,0), position) = offset; orient it via the
// owning Collider's localRotation. halfExtents span the local X/Z tangent axes of that plane.
struct PlaneCollider
{
    static constexpr ColliderShapeType type = ColliderShapeType::Plane;
    float     offset = 0.0f;
    glm::vec2 halfExtents{5.0f, 5.0f};
};

struct BoxCollider
{
    static constexpr ColliderShapeType type = ColliderShapeType::Box;
    glm::vec3 halfExtents{0.5f};
};

using ColliderShape = std::variant<SphereCollider, PlaneCollider, BoxCollider>;

inline ColliderShapeType colliderShapeType(const ColliderShape &shape)
{
    return std::visit([](const auto &value) { return std::decay_t<decltype(value)>::type; }, shape);
}

inline ColliderShape makeColliderShape(ColliderShapeType type)
{
    switch (type)
    {
        case ColliderShapeType::Sphere: return SphereCollider{};
        case ColliderShapeType::Plane: return PlaneCollider{};
        case ColliderShapeType::Box: return BoxCollider{};
    }
    return SphereCollider{};
}

struct Collider
{
    ColliderShape shape{SphereCollider{}};

    // Shape pose relative to the owning SceneObject's transform.
    glm::vec3 localPosition{0.0f};
    glm::quat localRotation{1.0f, 0.0f, 0.0f, 0.0f};

    PhysicsMaterial material;
};

} // namespace lr
