#include "SceneSerializer.hpp"

#include "MeshComponent.hpp"
#include "SceneAssets.hpp"
#include "core/scene/serialization/ComponentCodec.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
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

// 2 — objects, meshes and materials are named by UUID rather than by a file-local index.
constexpr int kFormatVersion = 2;
constexpr std::array<char, 8> kFileMagic{'L', 'R', 'S', 'C', 'E', 'N', 'E', '\0'};
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
    std::span<const std::byte> bytes() const { return m_bytes; }
private:
    std::vector<std::byte> m_bytes;
};

class BinaryReader
{
public:
    explicit BinaryReader(const std::filesystem::path &path, std::streamoff offset)
    {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("SceneSerializer: cannot open '" + path.string() + "'");
        const auto size = input.tellg();
        if (size < 0) throw std::runtime_error("SceneSerializer: cannot determine size of '" + path.string() + "'");
        if (offset < 0 || offset > size)
            throw std::runtime_error("SceneSerializer: invalid asset offset in '" + path.string() + "'");
        const auto payloadSize = size - offset;
        m_bytes.resize(static_cast<size_t>(payloadSize));
        input.seekg(offset);
        input.read(reinterpret_cast<char *>(m_bytes.data()), payloadSize);
        if (!input && payloadSize != 0) throw std::runtime_error("SceneSerializer: failed while reading '" + path.string() + "'");
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

json vec3(const glm::vec3 &value) { return {value.x, value.y, value.z}; }
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

json writeId(const Uuid &value) { return toString(value); }

Uuid readId(const json &value, const std::string &where)
{
    if (!value.is_string())
    {
        throw std::runtime_error("SceneSerializer: " + where + " must be a UUID string");
    }
    try
    {
        return parseUuid(value.get<std::string>());
    }
    catch (const std::runtime_error &error)
    {
        throw std::runtime_error("SceneSerializer: " + where + ": " + error.what());
    }
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
    save(assets.scene, assets.meshes, assets.materials, path);
}

void SceneSerializer::save(const Scene &scene, const MeshStore &meshStore,
                           const MaterialStore &materialStore, const std::filesystem::path &path)
{
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error("SceneSerializer: only little-endian hosts are currently supported");

    BinaryWriter binary;
    // Only assets a live object actually references are written. Collecting them in encounter order
    // keeps the file deterministic; the assets carry their own identities, so there is nothing to
    // assign here beyond deciding what to include.
    std::vector<MeshHandle>     referencedMeshes;
    std::vector<MaterialHandle> referencedMaterials;
    for (const auto &objectPointer : scene.sceneObjects())
    {
        if (!scene.contains(objectPointer->id()) || !objectPointer->hasComponent<MeshComponent>()) continue;
        const MeshComponent &component = objectPointer->getComponent<MeshComponent>();
        if (std::ranges::find(referencedMeshes, component.meshHandle()) == referencedMeshes.end())
            referencedMeshes.push_back(component.meshHandle());
        for (MaterialHandle handle : component.materialHandles())
        {
            if (std::ranges::find(referencedMaterials, handle) == referencedMaterials.end())
                referencedMaterials.push_back(handle);
        }
    }

    json materials = json::array();
    for (MaterialHandle handle : referencedMaterials)
    {
        json item  = serializeMaterial(materialStore.get(handle), binary);
        item["id"] = writeId(materialStore.idOf(handle));
        materials.push_back(std::move(item));
    }

    json meshes = json::array();
    for (MeshHandle handle : referencedMeshes)
    {
        const Mesh &mesh = meshStore.get(handle);
        json item{{"id", writeId(meshStore.idOf(handle))}, {"positions", binary.append(std::span(mesh.positions()))},
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

    const ComponentCodecRegistry codecs = makeSceneComponentCodecs();
    const ComponentSaveContext componentContext{meshStore, materialStore};
    json objects = json::array();
    for (const auto &objectPointer : scene.sceneObjects())
    {
        const SceneObject &object = *objectPointer;
        if (!scene.contains(object.id())) continue;
        json components = json::object();
        for (const std::type_index type : object.componentTypes())
        {
            const ComponentCodec &codec = codecs.forType(type);
            components[std::string(codec.key())] = codec.encode(object, componentContext);
        }
        json serialized{{"id", writeId(object.id())}, {"name", object.name}, {"parent", nullptr},
                        {"components", std::move(components)}};
        if (object.parent()) serialized["parent"] = writeId(*object.parent());
        objects.push_back(std::move(serialized));
    }

    json environment{{"hdri", nullptr}};
    if (scene.hdriPath())
    {
        std::vector<std::byte> encoded = scene.hdriData();
        if (encoded.empty())
        {
            std::ifstream hdri(*scene.hdriPath(), std::ios::binary | std::ios::ate);
            if (!hdri) throw std::runtime_error("SceneSerializer: cannot open HDRI '" + scene.hdriPath()->string() + "'");
            const std::streamsize size = hdri.tellg();
            if (size <= 0) throw std::runtime_error("SceneSerializer: HDRI is empty or unreadable: '" + scene.hdriPath()->string() + "'");
            encoded.resize(static_cast<size_t>(size));
            hdri.seekg(0);
            hdri.read(reinterpret_cast<char *>(encoded.data()), size);
            if (!hdri) throw std::runtime_error("SceneSerializer: failed while reading HDRI '" + scene.hdriPath()->string() + "'");
        }
        environment["hdri"] = {{"name", scene.hdriPath()->filename().generic_string()},
                               {"data", binary.appendBytes(encoded)}};
    }
    const json document{{"format", "lr.scene"}, {"version", kFormatVersion},
                        {"environment", std::move(environment)}, {"objects", std::move(objects)},
                        {"meshes", std::move(meshes)}, {"materials", std::move(materials)}};
    const std::string manifest = document.dump();
    const uint64_t manifestBytes = static_cast<uint64_t>(manifest.size());
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("SceneSerializer: cannot open '" + path.string() + "' for writing");
    output.write(kFileMagic.data(), static_cast<std::streamsize>(kFileMagic.size()));
    output.write(reinterpret_cast<const char *>(&manifestBytes), sizeof(manifestBytes));
    output.write(manifest.data(), static_cast<std::streamsize>(manifest.size()));
    const auto payload = binary.bytes();
    output.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
    if (!output) throw std::runtime_error("SceneSerializer: failed while writing '" + path.string() + "'");
}

std::unique_ptr<SceneAssets> SceneSerializer::load(const std::filesystem::path &path)
{
    auto result = std::make_unique<SceneAssets>();
    load(path, result->scene, result->meshes, result->materials);
    return result;
}

std::vector<SceneObjectId> SceneSerializer::load(const std::filesystem::path &path, Scene &scene,
                                                 MeshStore &meshStore, MaterialStore &materialStore)
{
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error("SceneSerializer: only little-endian hosts are currently supported");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("SceneSerializer: cannot open '" + path.string() + "'");

    std::array<char, kFileMagic.size()> magic{};
    uint64_t manifestBytes = 0;
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    input.read(reinterpret_cast<char *>(&manifestBytes), sizeof(manifestBytes));
    if (!input || magic != kFileMagic)
        throw std::runtime_error("SceneSerializer: '" + path.string() + "' is not an lrscene file");
    std::string manifest(static_cast<size_t>(manifestBytes), '\0');
    input.read(manifest.data(), static_cast<std::streamsize>(manifest.size()));
    if (!input) throw std::runtime_error("SceneSerializer: truncated manifest in '" + path.string() + "'");

    json document;
    try { document = json::parse(manifest); }
    catch (const json::exception &error)
    {
        throw std::runtime_error("SceneSerializer: invalid manifest in '" + path.string() + "': " + error.what());
    }
    if (required(document, "format", "root").get<std::string>() != "lr.scene")
        throw std::runtime_error("SceneSerializer: root.format must be 'lr.scene'");
    const int version = required(document, "version", "root").get<int>();
    if (version != kFormatVersion)
        throw std::runtime_error("SceneSerializer: unsupported format version " + std::to_string(version));
    const json &objects = required(document, "objects", "root");
    if (!objects.is_array()) throw std::runtime_error("SceneSerializer: root.objects must be an array");

    const std::streamoff payloadOffset = static_cast<std::streamoff>(kFileMagic.size() + sizeof(manifestBytes) + manifestBytes);
    const BinaryReader binary(path, payloadOffset);

    if (document.contains("environment"))
    {
        const json &environment = document.at("environment");
        const json &hdri = required(environment, "hdri", "root.environment");
        if (hdri.is_null()) scene.setHdriPath(std::nullopt);
        else if (hdri.is_object())
        {
            const std::string name = required(hdri, "name", "root.environment.hdri").get<std::string>();
            auto data = binary.read(required(hdri, "data", "root.environment.hdri"), "root.environment.hdri.data");
            scene.setEmbeddedHdri(std::filesystem::path(name), std::vector<std::byte>(data.begin(), data.end()));
        }
        else throw std::runtime_error("SceneSerializer: root.environment.hdri must be an object or null");
    }
    else scene.setHdriPath(std::nullopt);

    const json &materials = required(document, "materials", "root");
    if (!materials.is_array()) throw std::runtime_error("SceneSerializer: root.materials must be an array");
    for (size_t index = 0; index < materials.size(); ++index)
    {
        const std::string where = "materials[" + std::to_string(index) + "]";
        const MaterialId  id    = readId(required(materials[index], "id", where), where + ".id");
        materialStore.acquire(deserializeMaterial(materials[index], where, binary), id);
    }

    const json &meshes = required(document, "meshes", "root");
    if (!meshes.is_array()) throw std::runtime_error("SceneSerializer: root.meshes must be an array");
    for (size_t index = 0; index < meshes.size(); ++index)
    {
        const json &value = meshes[index];
        const std::string where = "meshes[" + std::to_string(index) + "]";
        const MeshId      id    = readId(required(value, "id", where), where + ".id");
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
        meshStore.add(std::move(mesh), id);
    }

    // Objects are created under the identities the file gives them, so every reference below —
    // parents, animation targets, skin joints — resolves to the same object it named when saved.
    // Scene::createSceneObject throws if one of those identities is already taken.
    std::vector<SceneObjectId> created;
    created.reserve(objects.size());
    const ComponentCodecRegistry codecs = makeSceneComponentCodecs();
    ComponentLoadContext componentContext{scene, meshStore, materialStore};
    for (size_t index = 0; index < objects.size(); ++index)
    {
        const json &value = objects[index];
        const std::string where = "objects[" + std::to_string(index) + "]";
        const SceneObjectId id = readId(required(value, "id", where), where + ".id");
        SceneObject &object = scene.createSceneObject(id);
        object.name = required(value, "name", where).get<std::string>();
        created.push_back(object.id());
    }

    for (size_t index = 0; index < objects.size(); ++index)
    {
        const json &value = objects[index];
        const std::string where = "objects[" + std::to_string(index) + "]";
        SceneObject &object = scene.getSceneObject(created[index]);
        const json &components = required(value, "components", where);
        if (!components.is_object()) throw std::runtime_error("SceneSerializer: " + where + ".components must be an object");
        for (const auto &[name, component] : components.items())
        {
            try
            {
                codecs.forKey(name).decode(component, object, componentContext,
                                           where + ".components." + name);
            }
            catch (const std::runtime_error &error)
            {
                if (std::string_view(error.what()).starts_with("SceneSerializer: unknown component"))
                    throw std::runtime_error(std::string(error.what()) + " at " + where + ".components");
                throw;
            }
        }

        const json &parent = required(value, "parent", where);
        if (!parent.is_null())
        {
            const SceneObjectId parentId = readId(parent, where + ".parent");
            if (!scene.contains(parentId)) throw std::runtime_error("SceneSerializer: missing parent object " + toString(parentId) + " at " + where);
            scene.setParent(object.id(), parentId);
        }
    }

    // Every object and component in the file now exists, so components may reach their siblings.
    for (SceneObjectId id : created)
    {
        scene.getSceneObject(id).onLoaded();
    }
    return created;
}

} // namespace lr
