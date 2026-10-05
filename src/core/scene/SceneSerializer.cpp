#include "SceneSerializer.hpp"

#include "Camera.hpp"
#include "Light.hpp"
#include "SceneAssets.hpp"
#include "TransformComponent.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace lr
{
namespace
{
using json = nlohmann::json;

constexpr int kFormatVersion = 1;

std::filesystem::path manifestPath(const std::filesystem::path &path) { return path / "scene.json"; }

json vec2(const glm::vec2 &value) { return {value.x, value.y}; }
json vec3(const glm::vec3 &value) { return {value.x, value.y, value.z}; }
json quat(const glm::quat &value) { return {value.x, value.y, value.z, value.w}; }

template <glm::length_t N, typename T, glm::qualifier Q>
glm::vec<N, T, Q> readVector(const json &value, const std::string &where)
{
    if (!value.is_array() || value.size() != N)
    {
        throw std::runtime_error("SceneSerializer: " + where + " must be an array of " + std::to_string(N) +
                                 " numbers");
    }
    glm::vec<N, T, Q> result{};
    for (glm::length_t i = 0; i < N; ++i)
    {
        if (!value[i].is_number())
        {
            throw std::runtime_error("SceneSerializer: " + where + " must contain only numbers");
        }
        result[i] = value[i].get<T>();
    }
    return result;
}

const json &required(const json &object, const char *key, const std::string &where)
{
    if (!object.is_object() || !object.contains(key))
    {
        throw std::runtime_error("SceneSerializer: missing " + where + "." + key);
    }
    return object.at(key);
}

json serializeLight(const LightVariant &variant)
{
    return std::visit(
        [](const auto &light) {
            json result{{"color", vec3(light.color)}, {"intensity", light.intensity}};
            using T = std::decay_t<decltype(light)>;
            if constexpr (std::is_same_v<T, PointLight>) result["type"] = "point";
            else if constexpr (std::is_same_v<T, SpotLight>)
            {
                result["type"] = "spot";
                result["inner_cone_degrees"] = light.innerConeAngleDegrees;
                result["outer_cone_degrees"] = light.outerConeAngleDegrees;
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
        },
        variant);
}

LightVariant deserializeLight(const json &value, const std::string &where)
{
    const std::string type = required(value, "type", where).get<std::string>();
    const glm::vec3 color = readVector<3, float, glm::defaultp>(required(value, "color", where), where + ".color");
    const float intensity = required(value, "intensity", where).get<float>();
    if (type == "point") return PointLight{{color, intensity}};
    if (type == "directional") return DirectionalLight{{color, intensity}};
    if (type == "image") return ImageLight{{color, intensity}};
    if (type == "spot")
    {
        SpotLight result{{color, intensity}};
        result.innerConeAngleDegrees = required(value, "inner_cone_degrees", where).get<float>();
        result.outerConeAngleDegrees = required(value, "outer_cone_degrees", where).get<float>();
        return result;
    }
    if (type == "area")
    {
        AreaLight result{{color, intensity}};
        result.size = readVector<2, float, glm::defaultp>(required(value, "size", where), where + ".size");
        result.twoSided = required(value, "two_sided", where).get<bool>();
        return result;
    }
    throw std::runtime_error("SceneSerializer: unknown light type '" + type + "' at " + where + ".type");
}

} // namespace

void SceneSerializer::save(const SceneAssets &assets, const std::filesystem::path &path)
{
    json objects = json::array();
    for (const auto &objectPointer : assets.scene.sceneObjects())
    {
        const SceneObject &object = *objectPointer;
        if (!assets.scene.contains(object.id())) continue;
        for (const std::type_index type : object.componentTypes())
        {
            if (type != typeid(TransformComponent) && type != typeid(Camera) && type != typeid(Light))
            {
                throw std::runtime_error("SceneSerializer: object '" + object.name + "' has unsupported component '" +
                                         std::string(type.name()) + "' in format version 1");
            }
        }

        json components = json::object();
        if (object.hasComponent<TransformComponent>())
        {
            const Transform &transform = object.getComponent<TransformComponent>().transform();
            components["transform"] = {{"position", vec3(transform.position())},
                                       {"rotation_xyzw", quat(transform.rotation())},
                                       {"scale", vec3(transform.scale())}};
        }
        if (object.hasComponent<Camera>())
        {
            const Camera &camera = object.getComponent<Camera>();
            components["camera"] = {{"projection", camera.projectionType == ProjectionType::Perspective
                                                           ? "perspective" : "orthographic"},
                                    {"fov_y_degrees", camera.fovYDegrees}, {"near_plane", camera.nearPlane},
                                    {"far_plane", camera.farPlane}, {"ortho_height", camera.orthoHeight}};
        }
        if (object.hasComponent<Light>())
        {
            components["light"] = serializeLight(object.getComponent<Light>().light);
        }
        json serialized{{"id", object.id()}, {"name", object.name}, {"parent", nullptr},
                        {"components", std::move(components)}};
        if (object.parent()) serialized["parent"] = *object.parent();
        objects.push_back(std::move(serialized));
    }

    const json document{{"format", "lr.scene"}, {"version", kFormatVersion}, {"objects", std::move(objects)}};
    std::filesystem::create_directories(path);
    const auto outputPath = manifestPath(path);
    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("SceneSerializer: cannot open '" + outputPath.string() + "' for writing");
    output << document.dump(2) << '\n';
    if (!output) throw std::runtime_error("SceneSerializer: failed while writing '" + outputPath.string() + "'");
}

std::unique_ptr<SceneAssets> SceneSerializer::load(const std::filesystem::path &path)
{
    const auto inputPath = manifestPath(path);
    std::ifstream input(inputPath, std::ios::binary);
    if (!input) throw std::runtime_error("SceneSerializer: cannot open '" + inputPath.string() + "'");

    json document;
    try { input >> document; }
    catch (const json::exception &error)
    {
        throw std::runtime_error("SceneSerializer: invalid JSON in '" + inputPath.string() + "': " + error.what());
    }
    if (required(document, "format", "root").get<std::string>() != "lr.scene")
        throw std::runtime_error("SceneSerializer: root.format must be 'lr.scene'");
    const int version = required(document, "version", "root").get<int>();
    if (version != kFormatVersion)
        throw std::runtime_error("SceneSerializer: unsupported format version " + std::to_string(version));
    const json &objects = required(document, "objects", "root");
    if (!objects.is_array()) throw std::runtime_error("SceneSerializer: root.objects must be an array");

    auto result = std::make_unique<SceneAssets>();
    std::unordered_map<SceneObjectId, SceneObjectId> ids;
    for (size_t index = 0; index < objects.size(); ++index)
    {
        const json &value = objects[index];
        const std::string where = "objects[" + std::to_string(index) + "]";
        const SceneObjectId fileId = required(value, "id", where).get<SceneObjectId>();
        if (ids.contains(fileId)) throw std::runtime_error("SceneSerializer: duplicate object id " + std::to_string(fileId));
        SceneObject &object = result->scene.createSceneObject();
        object.name = required(value, "name", where).get<std::string>();
        ids.emplace(fileId, object.id());
    }

    for (size_t index = 0; index < objects.size(); ++index)
    {
        const json &value = objects[index];
        const std::string where = "objects[" + std::to_string(index) + "]";
        const SceneObjectId fileId = required(value, "id", where).get<SceneObjectId>();
        SceneObject &object = result->scene.getSceneObject(ids.at(fileId));
        const json &components = required(value, "components", where);
        if (!components.is_object()) throw std::runtime_error("SceneSerializer: " + where + ".components must be an object");
        for (const auto &[name, ignored] : components.items())
        {
            if (name != "transform" && name != "camera" && name != "light")
                throw std::runtime_error("SceneSerializer: unknown component '" + name + "' at " + where + ".components");
        }
        if (components.contains("transform"))
        {
            const json &transform = components.at("transform");
            const glm::vec3 position = readVector<3, float, glm::defaultp>(required(transform, "position", where), where + ".components.transform.position");
            const glm::vec4 xyzw = readVector<4, float, glm::defaultp>(required(transform, "rotation_xyzw", where), where + ".components.transform.rotation_xyzw");
            const glm::vec3 scale = readVector<3, float, glm::defaultp>(required(transform, "scale", where), where + ".components.transform.scale");
            object.addComponent<TransformComponent>(position, glm::quat(xyzw.w, xyzw.x, xyzw.y, xyzw.z), scale);
        }
        if (components.contains("camera"))
        {
            const json &valueCamera = components.at("camera");
            Camera &camera = object.addComponent<Camera>();
            const std::string projection = required(valueCamera, "projection", where).get<std::string>();
            if (projection == "perspective") camera.projectionType = ProjectionType::Perspective;
            else if (projection == "orthographic") camera.projectionType = ProjectionType::Orthographic;
            else throw std::runtime_error("SceneSerializer: invalid camera projection at " + where);
            camera.fovYDegrees = required(valueCamera, "fov_y_degrees", where).get<float>();
            camera.nearPlane = required(valueCamera, "near_plane", where).get<float>();
            camera.farPlane = required(valueCamera, "far_plane", where).get<float>();
            camera.orthoHeight = required(valueCamera, "ortho_height", where).get<float>();
        }
        if (components.contains("light")) object.addComponent<Light>(deserializeLight(components.at("light"), where + ".components.light"));

        const json &parent = required(value, "parent", where);
        if (!parent.is_null())
        {
            const SceneObjectId parentFileId = parent.get<SceneObjectId>();
            const auto found = ids.find(parentFileId);
            if (found == ids.end()) throw std::runtime_error("SceneSerializer: missing parent object " + std::to_string(parentFileId) + " at " + where);
            result->scene.setParent(object.id(), found->second);
        }
    }
    return result;
}

} // namespace lr
