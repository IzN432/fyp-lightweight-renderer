#include "ComponentCodec.hpp"
#include "JsonUtils.hpp"

#include "core/editor/camera/SphericalCameraController.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <memory>
#include <stdexcept>

namespace lr
{
namespace
{
using namespace scene_serialization;

template <typename T> class TypedCodec : public ComponentCodec
{
public:
    std::type_index componentType() const final { return typeid(T); }
protected:
    static const T &component(const SceneObject &object) { return object.getComponent<T>(); }
};

class TransformCodec final : public TypedCodec<TransformComponent>
{
public:
    std::string_view key() const override { return "transform"; }
    json encode(const SceneObject &object, const ComponentSaveContext &) const override
    {
        const Transform &transform = component(object).transform();
        return {{"position", vec3(transform.position())},
                {"rotation_xyzw", quat(transform.rotation())}, {"scale", vec3(transform.scale())}};
    }
    void decode(const json &value, SceneObject &object, ComponentLoadContext &,
                const std::string &where) const override
    {
        const glm::vec3 position = readVector<3, float>(required(value, "position", where), where + ".position");
        const glm::vec4 xyzw = readVector<4, float>(required(value, "rotation_xyzw", where), where + ".rotation_xyzw");
        const glm::vec3 scale = readVector<3, float>(required(value, "scale", where), where + ".scale");
        object.addComponent<TransformComponent>(position, glm::quat(xyzw.w, xyzw.x, xyzw.y, xyzw.z), scale);
    }
};

class CameraCodec final : public TypedCodec<Camera>
{
public:
    std::string_view key() const override { return "camera"; }
    json encode(const SceneObject &object, const ComponentSaveContext &) const override
    {
        const Camera &camera = component(object);
        return {{"projection", camera.projectionType == ProjectionType::Perspective ? "perspective" : "orthographic"},
                {"fov_y_degrees", camera.fovYDegrees}, {"near_plane", camera.nearPlane},
                {"far_plane", camera.farPlane}, {"ortho_height", camera.orthoHeight}};
    }
    void decode(const json &value, SceneObject &object, ComponentLoadContext &,
                const std::string &where) const override
    {
        Camera &camera = object.addComponent<Camera>();
        const std::string projection = required(value, "projection", where).get<std::string>();
        if (projection == "perspective") camera.projectionType = ProjectionType::Perspective;
        else if (projection == "orthographic") camera.projectionType = ProjectionType::Orthographic;
        else throw std::runtime_error("SceneSerializer: invalid camera projection at " + where);
        camera.fovYDegrees = required(value, "fov_y_degrees", where).get<float>();
        camera.nearPlane = required(value, "near_plane", where).get<float>();
        camera.farPlane = required(value, "far_plane", where).get<float>();
        camera.orthoHeight = required(value, "ortho_height", where).get<float>();
    }
};

class SphericalCameraControllerCodec final : public TypedCodec<SphericalCameraController>
{
public:
    std::string_view key() const override { return "spherical_camera_controller"; }
    json encode(const SceneObject &object, const ComponentSaveContext &) const override
    {
        const auto state = component(object).orbitState();
        return {{"target", vec3(state.target)}, {"radius", state.radius}, {"azimuth", state.azimuth},
                {"elevation", state.elevation}};
    }
    void decode(const json &value, SceneObject &object, ComponentLoadContext &,
                const std::string &where) const override
    {
        SphericalCameraController::OrbitState state;
        state.target = readVector<3, float>(required(value, "target", where), where + ".target");
        state.radius = required(value, "radius", where).get<float>();
        state.azimuth = required(value, "azimuth", where).get<float>();
        state.elevation = required(value, "elevation", where).get<float>();
        object.addComponent<SphericalCameraController>().restoreOrbitState(state);
    }
};

json encodeLight(const LightVariant &variant)
{
    return std::visit([](const auto &light) {
        json result{{"color", vec3(light.color)}, {"intensity", light.intensity}};
        using T = std::decay_t<decltype(light)>;
        if constexpr (std::is_same_v<T, PointLight>) result["type"] = "point";
        else if constexpr (std::is_same_v<T, SpotLight>)
        {
            result["type"] = "spot";
            result["inner_cone_degrees"] = light.innerConeAngleDegrees;
            result["outer_cone_degrees"] = light.outerConeAngleDegrees;
            result["range"] = light.range;
            result["shadow_near_plane"] = light.shadowNearPlane;
        }
        else if constexpr (std::is_same_v<T, AreaLight>)
        {
            result["type"] = "area";
            result["size"] = vec2(light.size);
            result["two_sided"] = light.twoSided;
        }
        else if constexpr (std::is_same_v<T, DirectionalLight>) result["type"] = "directional";
        else result["type"] = "image";
        return result;
    }, variant);
}

LightVariant decodeLight(const json &value, const std::string &where)
{
    const std::string type = required(value, "type", where).get<std::string>();
    const glm::vec3 color = readVector<3, float>(required(value, "color", where), where + ".color");
    const float intensity = required(value, "intensity", where).get<float>();
    if (type == "point") return PointLight{{color, intensity}};
    if (type == "directional") return DirectionalLight{{color, intensity}};
    if (type == "image") return ImageLight{{color, intensity}};
    if (type == "spot")
    {
        SpotLight result{{color, intensity}};
        result.innerConeAngleDegrees = required(value, "inner_cone_degrees", where).get<float>();
        result.outerConeAngleDegrees = required(value, "outer_cone_degrees", where).get<float>();
        result.range = value.value("range", 100.0f);
        result.shadowNearPlane = value.value("shadow_near_plane", 0.1f);
        return result;
    }
    if (type == "area")
    {
        AreaLight result{{color, intensity}};
        result.size = readVector<2, float>(required(value, "size", where), where + ".size");
        result.twoSided = required(value, "two_sided", where).get<bool>();
        return result;
    }
    throw std::runtime_error("SceneSerializer: unknown light type '" + type + "' at " + where + ".type");
}

class LightCodec final : public TypedCodec<Light>
{
public:
    std::string_view key() const override { return "light"; }
    json encode(const SceneObject &object, const ComponentSaveContext &) const override
    {
        return encodeLight(component(object).light);
    }
    void decode(const json &value, SceneObject &object, ComponentLoadContext &,
                const std::string &where) const override
    {
        object.addComponent<Light>(decodeLight(value, where));
    }
};

class MeshComponentCodec final : public TypedCodec<MeshComponent>
{
public:
    std::string_view key() const override { return "mesh"; }
    // The component's material list is derived from the mesh's faceGroups, which are saved and
    // remapped with the mesh itself, so there is nothing material-side to write here. Files written
    // before that still carry a "materials" array; it is redundant now and simply ignored on load.
    json encode(const SceneObject &object, const ComponentSaveContext &context) const override
    {
        const MeshComponent &mesh = component(object);
        return {{"mesh", toString(context.meshes.idOf(mesh.meshHandle()))}};
    }
    void decode(const json &value, SceneObject &object, ComponentLoadContext &context,
                const std::string &where) const override
    {
        const MeshId meshId = readId(required(value, "mesh", where), where + ".mesh");
        const auto foundMesh = context.meshes.find(meshId);
        if (!foundMesh)
            throw std::runtime_error("SceneSerializer: missing mesh " + toString(meshId) + " at " + where);
        object.addComponent<MeshComponent>(*foundMesh, context.meshes, context.materials);
    }
};
} // namespace

void addCoreComponentCodecs(ComponentCodecRegistry &registry)
{
    registry.add(std::make_unique<TransformCodec>());
    registry.add(std::make_unique<CameraCodec>());
    registry.add(std::make_unique<SphericalCameraControllerCodec>());
    registry.add(std::make_unique<LightCodec>());
    registry.add(std::make_unique<MeshComponentCodec>());
}

} // namespace lr
