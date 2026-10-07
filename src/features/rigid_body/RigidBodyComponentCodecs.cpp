#include "core/scene/serialization/ComponentCodec.hpp"
#include "core/scene/serialization/JsonUtils.hpp"

#include "core/scene/SceneObject.hpp"
#include "features/rigid_body/ColliderComponent.hpp"
#include "features/rigid_body/RigidBodyComponent.hpp"

#include <memory>
#include <stdexcept>

namespace lr
{
namespace
{
using namespace scene_serialization;

json encodeCollider(const Collider &collider)
{
    json shape = std::visit([](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SphereCollider>)
            return json{{"type", "sphere"}, {"radius", value.radius}};
        else if constexpr (std::is_same_v<T, PlaneCollider>)
            return json{{"type", "plane"}, {"offset", value.offset}, {"half_extents", vec2(value.halfExtents)}};
        else return json{{"type", "box"}, {"half_extents", vec3(value.halfExtents)}};
    }, collider.shape);
    return {{"shape", std::move(shape)}, {"local_position", vec3(collider.localPosition)},
            {"local_rotation_xyzw", quat(collider.localRotation)},
            {"material", {{"restitution", collider.material.restitution},
                           {"friction", collider.material.friction}}}};
}

Collider decodeCollider(const json &value, const std::string &where)
{
    Collider result;
    const json &shape = required(value, "shape", where);
    const std::string type = required(shape, "type", where).get<std::string>();
    if (type == "sphere") result.shape = SphereCollider{required(shape, "radius", where).get<float>()};
    else if (type == "plane")
        result.shape = PlaneCollider{required(shape, "offset", where).get<float>(),
                                     readVector<2, float>(required(shape, "half_extents", where), where)};
    else if (type == "box")
        result.shape = BoxCollider{readVector<3, float>(required(shape, "half_extents", where), where)};
    else throw std::runtime_error("SceneSerializer: invalid collider shape '" + type + "' at " + where);
    result.localPosition = readVector<3, float>(required(value, "local_position", where), where);
    const glm::vec4 rotation = readVector<4, float>(required(value, "local_rotation_xyzw", where), where);
    result.localRotation = glm::quat(rotation.w, rotation.x, rotation.y, rotation.z);
    const json &material = required(value, "material", where);
    result.material.restitution = required(material, "restitution", where).get<float>();
    result.material.friction = required(material, "friction", where).get<float>();
    return result;
}

class RigidBodyCodec final : public ComponentCodec
{
public:
    std::string_view key() const override { return "rigid_body"; }
    std::type_index componentType() const override { return typeid(RigidBodyComponent); }
    json encode(const SceneObject &object, const ComponentSaveContext &) const override
    {
        const RigidBodyComponent &body = object.getComponent<RigidBodyComponent>();
        return {{"type", body.isStatic() ? "static" : "dynamic"}, {"mass", body.mass()},
                {"inertia_diagonal", vec3(body.inertiaDiagonal())}, {"linear_drag", body.linearDrag()},
                {"angular_drag", body.angularDrag()}, {"linear_velocity", vec3(body.linearVelocity())},
                {"angular_velocity", vec3(body.angularVelocity())}};
    }
    void decode(const json &value, SceneObject &object, ComponentLoadContext &,
                const std::string &where) const override
    {
        const std::string type = required(value, "type", where).get<std::string>();
        if (type != "static" && type != "dynamic")
            throw std::runtime_error("SceneSerializer: invalid rigid body type at " + where);
        auto &body = object.addComponent<RigidBodyComponent>(required(value, "mass", where).get<float>(),
            type == "static" ? RigidBodyType::Static : RigidBodyType::Dynamic);
        body.setInertiaDiagonal(readVector<3, float>(required(value, "inertia_diagonal", where), where));
        body.setLinearDrag(required(value, "linear_drag", where).get<float>());
        body.setAngularDrag(required(value, "angular_drag", where).get<float>());
        body.setLinearVelocity(readVector<3, float>(required(value, "linear_velocity", where), where));
        body.setAngularVelocity(readVector<3, float>(required(value, "angular_velocity", where), where));
    }
};

class ColliderCodec final : public ComponentCodec
{
public:
    std::string_view key() const override { return "colliders"; }
    std::type_index componentType() const override { return typeid(ColliderComponent); }
    json encode(const SceneObject &object, const ComponentSaveContext &) const override
    {
        json result = json::array();
        for (const Collider &collider : object.getComponent<ColliderComponent>().colliders())
            result.push_back(encodeCollider(collider));
        return result;
    }
    void decode(const json &value, SceneObject &object, ComponentLoadContext &,
                const std::string &where) const override
    {
        if (!value.is_array() || value.empty())
            throw std::runtime_error("SceneSerializer: colliders must be a non-empty array at " + where);
        std::vector<Collider> colliders;
        for (size_t i = 0; i < value.size(); ++i)
            colliders.push_back(decodeCollider(value[i], where + "[" + std::to_string(i) + "]"));
        object.addComponent<ColliderComponent>(std::move(colliders));
    }
};
} // namespace

void addRigidBodyComponentCodecs(ComponentCodecRegistry &registry)
{
    registry.add(std::make_unique<RigidBodyCodec>());
    registry.add(std::make_unique<ColliderCodec>());
}

} // namespace lr
