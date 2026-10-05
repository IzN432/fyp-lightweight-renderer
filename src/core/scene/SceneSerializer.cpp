#include "SceneSerializer.hpp"

#include "Camera.hpp"
#include "Light.hpp"
#include "MeshComponent.hpp"
#include "SceneAssets.hpp"
#include "TransformComponent.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <bit>
#include <cstring>
#include <set>
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
std::filesystem::path assetPath(const std::filesystem::path &path) { return path / "assets.bin"; }
const json &required(const json &object, const char *key, const std::string &where);

class BinaryWriter
{
public:
    template <typename T> json append(std::span<const T> values)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        const uint64_t offset = static_cast<uint64_t>(m_bytes.size());
        const auto bytes = std::as_bytes(values);
        m_bytes.insert(m_bytes.end(), bytes.begin(), bytes.end());
        return {{"offset", offset}, {"bytes", bytes.size()}};
    }
    json appendBytes(std::span<const std::byte> values)
    {
        const uint64_t offset = static_cast<uint64_t>(m_bytes.size());
        m_bytes.insert(m_bytes.end(), values.begin(), values.end());
        return {{"offset", offset}, {"bytes", values.size()}};
    }
    void write(const std::filesystem::path &path) const
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("SceneSerializer: cannot open '" + path.string() + "' for writing");
        output.write(reinterpret_cast<const char *>(m_bytes.data()), static_cast<std::streamsize>(m_bytes.size()));
        if (!output) throw std::runtime_error("SceneSerializer: failed while writing '" + path.string() + "'");
    }
private:
    std::vector<std::byte> m_bytes;
};

class BinaryReader
{
public:
    explicit BinaryReader(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("SceneSerializer: cannot open '" + path.string() + "'");
        const auto size = input.tellg();
        if (size < 0) throw std::runtime_error("SceneSerializer: cannot determine size of '" + path.string() + "'");
        m_bytes.resize(static_cast<size_t>(size));
        input.seekg(0);
        input.read(reinterpret_cast<char *>(m_bytes.data()), size);
        if (!input && size != 0) throw std::runtime_error("SceneSerializer: failed while reading '" + path.string() + "'");
    }
    std::span<const std::byte> read(const json &view, const std::string &where) const
    {
        const uint64_t offset = required(view, "offset", where).get<uint64_t>();
        const uint64_t bytes = required(view, "bytes", where).get<uint64_t>();
        if (offset > m_bytes.size() || bytes > m_bytes.size() - offset)
            throw std::runtime_error("SceneSerializer: binary view is out of bounds at " + where);
        return std::span<const std::byte>(m_bytes).subspan(static_cast<size_t>(offset), static_cast<size_t>(bytes));
    }
    template <typename T> std::vector<T> readArray(const json &view, const std::string &where) const
    {
        const auto bytes = read(view, where);
        if (bytes.size() % sizeof(T) != 0)
            throw std::runtime_error("SceneSerializer: binary byte count has the wrong stride at " + where);
        std::vector<T> result(bytes.size() / sizeof(T));
        std::memcpy(result.data(), bytes.data(), bytes.size());
        return result;
    }
private:
    std::vector<std::byte> m_bytes;
};

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

std::string attributeType(const MeshLayout::AttributeDesc &attribute)
{
    if (attribute.type == typeid(float) && attribute.stride == sizeof(float)) return "f32";
    if (attribute.type == typeid(uint32_t) && attribute.stride == sizeof(uint32_t)) return "u32";
    if (attribute.type == typeid(glm::vec2) && attribute.stride == sizeof(glm::vec2)) return "vec2f";
    if (attribute.type == typeid(glm::vec3) && attribute.stride == sizeof(glm::vec3)) return "vec3f";
    if (attribute.type == typeid(glm::vec4) && attribute.stride == sizeof(glm::vec4)) return "vec4f";
    throw std::runtime_error("SceneSerializer: unsupported mesh attribute type for '" + attribute.name + "'");
}

template <typename Getter>
json serializeAttributes(const std::vector<MeshLayout::AttributeDesc> &attributes, Getter get, BinaryWriter &binary)
{
    json result = json::array();
    for (const auto &attribute : attributes)
    {
        json item{{"name", attribute.name}, {"type", attributeType(attribute)}};
        item["data"] = binary.appendBytes(get(attribute.name));
        result.push_back(std::move(item));
    }
    return result;
}

template <typename Setter>
void deserializeAttributes(const json &attributes, const std::string &where, const BinaryReader &binary, Setter set)
{
    if (!attributes.is_array()) throw std::runtime_error("SceneSerializer: " + where + " must be an array");
    for (size_t i = 0; i < attributes.size(); ++i)
    {
        const json &item = attributes[i];
        const std::string itemWhere = where + "[" + std::to_string(i) + "]";
        const std::string name = required(item, "name", itemWhere).get<std::string>();
        const std::string type = required(item, "type", itemWhere).get<std::string>();
        const auto bytes = binary.read(required(item, "data", itemWhere), itemWhere + ".data");
        if (type == "f32") set.template operator()<float>(name, bytes);
        else if (type == "u32") set.template operator()<uint32_t>(name, bytes);
        else if (type == "vec2f") set.template operator()<glm::vec2>(name, bytes);
        else if (type == "vec3f") set.template operator()<glm::vec3>(name, bytes);
        else if (type == "vec4f") set.template operator()<glm::vec4>(name, bytes);
        else throw std::runtime_error("SceneSerializer: unsupported attribute type '" + type + "' at " + itemWhere);
    }
}

template <typename T> std::span<const T> typedBytes(std::span<const std::byte> bytes, const std::string &where)
{
    if (bytes.size() % sizeof(T) != 0)
        throw std::runtime_error("SceneSerializer: attribute data has the wrong stride at " + where);
    return {reinterpret_cast<const T *>(bytes.data()), bytes.size() / sizeof(T)};
}

json serializeMaterial(const Material &material, BinaryWriter &binary)
{
    json parameters = json::object();
    for (const auto &[name, value] : material.parameters)
    {
        parameters[name] = std::visit([](const auto &parameter) {
            using T = std::decay_t<decltype(parameter)>;
            if constexpr (std::is_same_v<T, MaterialParam::ColorRGBA>) return json{{"type", "rgba"}, {"value", {parameter.value.x, parameter.value.y, parameter.value.z, parameter.value.w}}};
            else if constexpr (std::is_same_v<T, MaterialParam::ColorRGB>) return json{{"type", "rgb"}, {"value", vec3(parameter.value)}};
            else if constexpr (std::is_same_v<T, MaterialParam::NormalizedFloat>) return json{{"type", "normalized"}, {"value", parameter.value}};
            else return json{{"type", "ranged"}, {"value", parameter.value}, {"floor", parameter.floor}, {"ceiling", parameter.ceiling}};
        }, value);
    }
    json textures = json::object();
    for (const auto &[name, image] : material.textures)
    {
        textures[name] = {{"name", image.name}, {"width", image.width}, {"height", image.height},
                          {"pixels", binary.append(std::span<const uint8_t>(image.pixels))}};
    }
    return {{"name", material.name}, {"parameters", std::move(parameters)}, {"textures", std::move(textures)}};
}

Material deserializeMaterial(const json &value, const std::string &where, const BinaryReader &binary)
{
    Material material;
    material.name = required(value, "name", where).get<std::string>();
    const json &parameters = required(value, "parameters", where);
    for (const auto &[name, parameter] : parameters.items())
    {
        const std::string type = required(parameter, "type", where + ".parameters." + name).get<std::string>();
        if (type == "rgba")
        {
            const glm::vec4 v = readVector<4, float, glm::defaultp>(required(parameter, "value", where), where);
            material.parameters[name] = MaterialParam::ColorRGBA{v};
        }
        else if (type == "rgb") material.parameters[name] = MaterialParam::ColorRGB{readVector<3, float, glm::defaultp>(required(parameter, "value", where), where)};
        else if (type == "normalized") material.parameters[name] = MaterialParam::NormalizedFloat{required(parameter, "value", where).get<float>()};
        else if (type == "ranged") material.parameters[name] = MaterialParam::RangedFloat{required(parameter, "value", where).get<float>(), required(parameter, "floor", where).get<float>(), required(parameter, "ceiling", where).get<float>()};
        else throw std::runtime_error("SceneSerializer: unknown material parameter type '" + type + "' at " + where);
    }
    const json &textures = required(value, "textures", where);
    for (const auto &[name, texture] : textures.items())
    {
        MaterialImage image;
        image.name = required(texture, "name", where).get<std::string>();
        image.width = required(texture, "width", where).get<uint32_t>();
        image.height = required(texture, "height", where).get<uint32_t>();
        image.pixels = binary.readArray<uint8_t>(required(texture, "pixels", where), where + ".textures." + name);
        const uint64_t expected = static_cast<uint64_t>(image.width) * image.height * 4;
        if (image.pixels.size() != expected) throw std::runtime_error("SceneSerializer: RGBA texture byte count mismatch at " + where + ".textures." + name);
        material.textures.emplace(name, std::move(image));
    }
    return material;
}

} // namespace

void SceneSerializer::save(const SceneAssets &assets, const std::filesystem::path &path)
{
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error("SceneSerializer: only little-endian hosts are currently supported");

    BinaryWriter binary;
    std::unordered_map<MeshHandle, uint32_t> meshIds;
    std::unordered_map<MaterialHandle, uint32_t> materialIds;
    for (const auto &objectPointer : assets.scene.sceneObjects())
    {
        if (!assets.scene.contains(objectPointer->id()) || !objectPointer->hasComponent<MeshComponent>()) continue;
        const MeshComponent &component = objectPointer->getComponent<MeshComponent>();
        meshIds.try_emplace(component.meshHandle(), static_cast<uint32_t>(meshIds.size()));
        for (MaterialHandle handle : component.materialHandles())
            materialIds.try_emplace(handle, static_cast<uint32_t>(materialIds.size()));
    }

    json materials = json::array();
    std::vector<std::pair<MaterialHandle, uint32_t>> orderedMaterials(materialIds.begin(), materialIds.end());
    std::ranges::sort(orderedMaterials, {}, &std::pair<MaterialHandle, uint32_t>::second);
    for (const auto &[handle, id] : orderedMaterials)
    {
        json item = serializeMaterial(assets.materials.get(handle), binary);
        item["id"] = id;
        materials.push_back(std::move(item));
    }

    json meshes = json::array();
    std::vector<std::pair<MeshHandle, uint32_t>> orderedMeshes(meshIds.begin(), meshIds.end());
    std::ranges::sort(orderedMeshes, {}, &std::pair<MeshHandle, uint32_t>::second);
    for (const auto &[handle, id] : orderedMeshes)
    {
        const Mesh &mesh = assets.meshes.get(handle);
        json item{{"id", id}, {"positions", binary.append(std::span(mesh.positions()))},
                  {"position_indices", binary.append(std::span(mesh.positionIndices()))},
                  {"faces", binary.append(std::span(mesh.faces()))},
                  {"face_groups", binary.append(std::span(mesh.faceGroups()))},
                  {"face_group_count", mesh.faceGroupCount()}, {"vertex_group_count", mesh.vertexGroupCount()}};
        item["per_vertex"] = serializeAttributes(mesh.layout().perVertexAttrs(), [&](const std::string &name) { return mesh.rawPerVertexData(name); }, binary);
        item["per_unique_vertex"] = serializeAttributes(mesh.layout().perUniqueVertexAttrs(), [&](const std::string &name) { return mesh.rawPerUniqueVertexData(name); }, binary);
        item["per_face"] = serializeAttributes(mesh.layout().perFaceAttrs(), [&](const std::string &name) { return mesh.rawPerFaceData(name); }, binary);
        item["face_group_attributes"] = serializeAttributes(mesh.layout().faceGroupAttrs(), [&](const std::string &name) { return mesh.rawFaceGroupAttributeData(name); }, binary);
        item["vertex_group_attributes"] = serializeAttributes(mesh.layout().vertexGroupAttrs(), [&](const std::string &name) { return mesh.rawVertexGroupAttributeData(name); }, binary);
        item["vertex_group_entries"] = binary.append(mesh.rawGroupEntries());
        item["vertex_group_offsets"] = binary.append(mesh.rawGroupOffsets());
        meshes.push_back(std::move(item));
    }

    json objects = json::array();
    for (const auto &objectPointer : assets.scene.sceneObjects())
    {
        const SceneObject &object = *objectPointer;
        if (!assets.scene.contains(object.id())) continue;
        for (const std::type_index type : object.componentTypes())
        {
            if (type != typeid(TransformComponent) && type != typeid(Camera) && type != typeid(Light) &&
                type != typeid(MeshComponent))
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
        if (object.hasComponent<MeshComponent>())
        {
            const MeshComponent &mesh = object.getComponent<MeshComponent>();
            json materialReferences = json::array();
            for (MaterialHandle handle : mesh.materialHandles()) materialReferences.push_back(materialIds.at(handle));
            components["mesh"] = {{"mesh", meshIds.at(mesh.meshHandle())},
                                  {"materials", std::move(materialReferences)}};
        }
        json serialized{{"id", object.id()}, {"name", object.name}, {"parent", nullptr},
                        {"components", std::move(components)}};
        if (object.parent()) serialized["parent"] = *object.parent();
        objects.push_back(std::move(serialized));
    }

    const json document{{"format", "lr.scene"}, {"version", kFormatVersion}, {"objects", std::move(objects)},
                        {"meshes", std::move(meshes)}, {"materials", std::move(materials)},
                        {"binary", "assets.bin"}};
    std::filesystem::create_directories(path);
    binary.write(assetPath(path));
    const auto outputPath = manifestPath(path);
    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("SceneSerializer: cannot open '" + outputPath.string() + "' for writing");
    output << document.dump(2) << '\n';
    if (!output) throw std::runtime_error("SceneSerializer: failed while writing '" + outputPath.string() + "'");
}

std::unique_ptr<SceneAssets> SceneSerializer::load(const std::filesystem::path &path)
{
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error("SceneSerializer: only little-endian hosts are currently supported");
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
    const BinaryReader binary(path / required(document, "binary", "root").get<std::string>());

    std::unordered_map<uint32_t, MaterialHandle> materialHandles;
    const json &materials = required(document, "materials", "root");
    if (!materials.is_array()) throw std::runtime_error("SceneSerializer: root.materials must be an array");
    for (size_t index = 0; index < materials.size(); ++index)
    {
        const std::string where = "materials[" + std::to_string(index) + "]";
        const uint32_t id = required(materials[index], "id", where).get<uint32_t>();
        if (materialHandles.contains(id)) throw std::runtime_error("SceneSerializer: duplicate material id " + std::to_string(id));
        materialHandles.emplace(id, result->materials.acquire(deserializeMaterial(materials[index], where, binary)));
    }

    std::unordered_map<uint32_t, MeshHandle> meshHandles;
    const json &meshes = required(document, "meshes", "root");
    if (!meshes.is_array()) throw std::runtime_error("SceneSerializer: root.meshes must be an array");
    for (size_t index = 0; index < meshes.size(); ++index)
    {
        const json &value = meshes[index];
        const std::string where = "meshes[" + std::to_string(index) + "]";
        const uint32_t id = required(value, "id", where).get<uint32_t>();
        if (meshHandles.contains(id)) throw std::runtime_error("SceneSerializer: duplicate mesh id " + std::to_string(id));
        Mesh mesh;
        auto positions = binary.readArray<glm::vec3>(required(value, "positions", where), where + ".positions");
        auto positionIndices = binary.readArray<uint32_t>(required(value, "position_indices", where), where + ".position_indices");
        auto faces = binary.readArray<glm::uvec3>(required(value, "faces", where), where + ".faces");
        mesh.setTopology(std::move(positions), std::move(positionIndices), std::move(faces));
        auto faceGroups = binary.readArray<uint32_t>(required(value, "face_groups", where), where + ".face_groups");
        if (!faceGroups.empty()) mesh.setFaceGroups(std::move(faceGroups));
        mesh.setFaceGroupCount(required(value, "face_group_count", where).get<uint32_t>());
        const uint32_t vertexGroupCount = required(value, "vertex_group_count", where).get<uint32_t>();

        auto loadDomain = [&](const char *key, auto setter) {
            deserializeAttributes(required(value, key, where), where + "." + key, binary, setter);
        };
        loadDomain("per_vertex", [&]<typename T>(const std::string &name, std::span<const std::byte> bytes) {
            mesh.setPerVertexArray<T>(name, typedBytes<T>(bytes, where + ".per_vertex." + name));
        });
        loadDomain("per_unique_vertex", [&]<typename T>(const std::string &name, std::span<const std::byte> bytes) {
            mesh.setPerUniqueVertexArray<T>(name, typedBytes<T>(bytes, where + ".per_unique_vertex." + name));
        });
        loadDomain("per_face", [&]<typename T>(const std::string &name, std::span<const std::byte> bytes) {
            mesh.setPerFaceArray<T>(name, typedBytes<T>(bytes, where + ".per_face." + name));
        });
        loadDomain("face_group_attributes", [&]<typename T>(const std::string &name, std::span<const std::byte> bytes) {
            mesh.setFaceGroupAttributeArray<T>(name, typedBytes<T>(bytes, where + ".face_group_attributes." + name));
        });

        const auto entries = binary.readArray<VertexGroupEntry>(required(value, "vertex_group_entries", where), where + ".vertex_group_entries");
        const auto offsets = binary.readArray<uint32_t>(required(value, "vertex_group_offsets", where), where + ".vertex_group_offsets");
        if (!offsets.empty())
        {
            if (offsets.size() != mesh.uniquePositionCount() + 1 || offsets.front() != 0 || offsets.back() != entries.size())
                throw std::runtime_error("SceneSerializer: invalid vertex-group offsets at " + where);
            mesh.enableVertexGroups();
            mesh.setVertexGroupCount(vertexGroupCount);
            for (uint32_t vertex = 0; vertex < mesh.uniquePositionCount(); ++vertex)
            {
                if (offsets[vertex] > offsets[vertex + 1] || offsets[vertex + 1] > entries.size())
                    throw std::runtime_error("SceneSerializer: invalid vertex-group range at " + where);
                mesh.setVertexGroups(vertex, std::span(entries).subspan(offsets[vertex], offsets[vertex + 1] - offsets[vertex]));
            }
        }
        loadDomain("vertex_group_attributes", [&]<typename T>(const std::string &name, std::span<const std::byte> bytes) {
            mesh.setVertexGroupAttributeArray<T>(name, typedBytes<T>(bytes, where + ".vertex_group_attributes." + name));
        });
        meshHandles.emplace(id, result->meshes.add(std::move(mesh)));
    }

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
            if (name != "transform" && name != "camera" && name != "light" && name != "mesh")
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
        if (components.contains("mesh"))
        {
            const json &mesh = components.at("mesh");
            const uint32_t meshId = required(mesh, "mesh", where).get<uint32_t>();
            const auto foundMesh = meshHandles.find(meshId);
            if (foundMesh == meshHandles.end()) throw std::runtime_error("SceneSerializer: missing mesh " + std::to_string(meshId) + " at " + where);
            std::vector<MaterialHandle> objectMaterials;
            const json &references = required(mesh, "materials", where);
            if (!references.is_array()) throw std::runtime_error("SceneSerializer: mesh materials must be an array at " + where);
            for (const json &reference : references)
            {
                const uint32_t materialId = reference.get<uint32_t>();
                const auto foundMaterial = materialHandles.find(materialId);
                if (foundMaterial == materialHandles.end()) throw std::runtime_error("SceneSerializer: missing material " + std::to_string(materialId) + " at " + where);
                objectMaterials.push_back(foundMaterial->second);
            }
            object.addComponent<MeshComponent>(foundMesh->second, result->meshes, std::move(objectMaterials),
                                               result->materials);
        }

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
