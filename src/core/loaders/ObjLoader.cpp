#include "core/loaders/ObjLoader.hpp"
#include "core/utility/ImageLoader.hpp"
#include "core/loaders/LoaderUtils.hpp"

#include <tiny_obj_loader.h>
#include <mikktspace.h>
#include <spdlog/spdlog.h>
namespace lr
{
namespace
{

MaterialImage loadMaterialImage(const std::filesystem::path &objDirectoryPath, const std::string &texName)
{
    if (texName.empty())
    {
        return {};
    }

    std::filesystem::path p(texName);
    if (!p.is_absolute())
    {
        p = (objDirectoryPath / p).lexically_normal();
    }

    if (!std::filesystem::exists(p))
    {
        return {};
    }

    LoadedImage loaded = loadImageFromFile(p);
    if (loaded.empty())
    {
        return {};
    }

    const size_t  byteCount = static_cast<size_t>(loaded.width) * loaded.height * 4;
    MaterialImage out;
    out.width  = loaded.width;
    out.height = loaded.height;
    out.pixels.assign(loaded.pixels, loaded.pixels + byteCount);
    return out;
}

// Registers each material into `materialStore` as it's built and returns the resulting handles,
// parallel to the OBJ's material indices (index 0 = MaterialStore's shared default material,
// index i+1 = materials[i]) — extractMeshData bakes these straight into each face's faceGroups entry.
std::vector<MaterialHandle> extractMaterials(const tinyobj::ObjReader    &reader,
                                             const std::filesystem::path &objDirectoryPath,
                                             const ObjLoaderConfig &config, MaterialStore &materialStore)
{
    std::vector<MaterialHandle> handles;

    const auto &materials = reader.GetMaterials();
    handles.reserve(materials.size() + 1);
    handles.push_back(materialStore.defaultMaterialHandle());

    // Fallback texture values used to patch any real material missing a required texture — just a
    // handful of 1x1 pixels, cheap to keep local rather than routing through the store.
    const MaterialImage fallbackDiffuse   = MaterialImage::singlePixel(glm::vec4(1.0f));
    const MaterialImage fallbackAmbient   = MaterialImage::singlePixel(glm::vec4(0.0f));
    const MaterialImage fallbackSpecular  = MaterialImage::singlePixel(glm::vec4(0.0f));
    const MaterialImage fallbackNormal    = MaterialImage::singlePixel(glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
    const MaterialImage fallbackMetallic  = MaterialImage::singlePixel(glm::vec4(0.0f));
    const MaterialImage fallbackRoughness = MaterialImage::singlePixel(glm::vec4(1.0f));
    const MaterialImage fallbackEmissive  = MaterialImage::singlePixel(glm::vec4(0.0f));

    for (const auto &m : materials)
    {
        Material mat;
        mat.name = m.name;
        mat.parameters[config.baseDiffuseName] =
            MaterialParam::ColorRGBA{glm::vec4(m.diffuse[0], m.diffuse[1], m.diffuse[2], 1.0f)};
        mat.parameters[config.baseAmbientName] =
            MaterialParam::ColorRGBA{glm::vec4(m.ambient[0], m.ambient[1], m.ambient[2], 1.0f)};
        mat.parameters[config.baseSpecularName] =
            MaterialParam::ColorRGBA{glm::vec4(m.specular[0], m.specular[1], m.specular[2], 1.0f)};
        mat.parameters[config.shininessName]     = MaterialParam::RangedFloat{m.shininess, 0.0f, 128.0f};
        mat.parameters[config.baseRoughnessName] = MaterialParam::NormalizedFloat{m.roughness};
        mat.parameters[config.baseMetallicName]  = MaterialParam::NormalizedFloat{m.metallic};
        mat.parameters[config.baseEmissiveName] =
            MaterialParam::ColorRGB{glm::vec3(m.emission[0], m.emission[1], m.emission[2])};
        mat.textures[config.diffuseTextureName]   = loadMaterialImage(objDirectoryPath, m.diffuse_texname);
        mat.textures[config.ambientTextureName]   = loadMaterialImage(objDirectoryPath, m.ambient_texname);
        mat.textures[config.specularTextureName]  = loadMaterialImage(objDirectoryPath, m.specular_texname);
        mat.textures[config.normalTextureName]    = loadMaterialImage(objDirectoryPath, m.normal_texname);
        mat.textures[config.metallicTextureName]  = loadMaterialImage(objDirectoryPath, m.metallic_texname);
        mat.textures[config.roughnessTextureName] = loadMaterialImage(objDirectoryPath, m.roughness_texname);
        mat.textures[config.emissiveTextureName]  = loadMaterialImage(objDirectoryPath, m.emissive_texname);

        if (mat.textures[config.diffuseTextureName].pixels.empty())
        {
            mat.textures[config.diffuseTextureName] = fallbackDiffuse;
        }
        if (mat.textures[config.ambientTextureName].pixels.empty())
        {
            mat.textures[config.ambientTextureName] = fallbackAmbient;
        }
        if (mat.textures[config.specularTextureName].pixels.empty())
        {
            mat.textures[config.specularTextureName] = fallbackSpecular;
        }
        if (mat.textures[config.normalTextureName].pixels.empty())
        {
            mat.textures[config.normalTextureName] = fallbackNormal;
        }
        if (mat.textures[config.metallicTextureName].pixels.empty())
        {
            mat.textures[config.metallicTextureName] = fallbackMetallic;
        }
        if (mat.textures[config.roughnessTextureName].pixels.empty())
        {
            mat.textures[config.roughnessTextureName] = fallbackRoughness;
        }
        if (mat.textures[config.emissiveTextureName].pixels.empty())
        {
            mat.textures[config.emissiveTextureName] = fallbackEmissive;
        }

        handles.push_back(materialStore.acquire(std::move(mat)));
    }

    return handles;
}

tinyobj::ObjReader loadObjFile(const std::filesystem::path &path)
{
    tinyobj::ObjReader       reader;
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;

    if (!reader.ParseFromFile(path.string(), config))
    {
        std::string msg = "ObjLoader: failed to parse '" + path.string() + "'";
        if (!reader.Error().empty())
        {
            msg += ": " + reader.Error();
        }
        throw std::runtime_error(msg);
    }

    if (!reader.Warning().empty())
    {
        spdlog::warn("ObjLoader warning while parsing '{}': {}", path.string(), reader.Warning());
    }

    return reader;
}

struct VertexKey
{
    int positionIdx = -1;
    int uvIdx       = -1;
    int normalIdx   = -1;

    bool operator==(const VertexKey &other) const
    {
        return positionIdx == other.positionIdx && uvIdx == other.uvIdx && normalIdx == other.normalIdx;
    }
};

struct VertexKeyHash
{
    std::size_t operator()(const VertexKey &key) const
    {
        std::size_t seed    = 0;
        auto        combine = [&](int val) {
            std::size_t h = std::hash<int>{}(val);
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };

        combine(key.positionIdx);
        combine(key.uvIdx);
        combine(key.normalIdx);
        return seed;
    }
};

struct Vector3Hash
{
    std::size_t operator()(const glm::vec3 &v) const
    {
        std::size_t seed    = 0;
        auto        combine = [&](float val) {
            std::size_t h = std::hash<float>{}(val);
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };

        combine(v.x);
        combine(v.y);
        combine(v.z);
        return seed;
    }
};

MeshData extractMeshData(const tinyobj::ObjReader &reader, const std::filesystem::path &path,
                         const std::vector<MaterialHandle> &materialHandles)
{
    MeshData meshData;
    auto    &positions       = meshData.positions;
    auto    &positionIndices = meshData.positionIndices;
    auto    &normals         = meshData.normals;
    auto    &tangents        = meshData.tangents;
    auto    &uvs             = meshData.uvs;
    auto    &faces           = meshData.faces;
    auto    &faceGroups      = meshData.faceGroups;

    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> vertexMap;

    const tinyobj::attrib_t &attributes = reader.GetAttrib();

    positions.reserve(attributes.vertices.size() / 3);
    normals.reserve(attributes.normals.size() / 3);
    uvs.reserve(attributes.texcoords.size() / 2);

    size_t indicesCount = 0;
    for (const auto &shape : reader.GetShapes())
    {
        indicesCount += shape.mesh.indices.size();
    }

    faces.reserve(indicesCount / 3);
    faceGroups.reserve(indicesCount / 3);

    std::unordered_map<glm::vec3, uint32_t, Vector3Hash> positionToIndex;

    auto getOrAddVertex = [&](const tinyobj::index_t &idx) -> uint32_t {
        VertexKey key{idx.vertex_index, idx.texcoord_index, idx.normal_index};
        auto [it, inserted] = vertexMap.try_emplace(key, static_cast<uint32_t>(positionIndices.size()));
        if (!inserted)
        {
            return it->second;
        }

        glm::vec3 position(attributes.vertices[3 * idx.vertex_index + 0], attributes.vertices[3 * idx.vertex_index + 1],
                           attributes.vertices[3 * idx.vertex_index + 2]);

        if (positionToIndex.find(position) == positionToIndex.end())
        {
            positionToIndex[position] = static_cast<uint32_t>(positions.size());
            positions.push_back(position);
            positionIndices.push_back(positionToIndex[position]);
        } else
        {
            // The position is already in the map, meaning this is a duplicate vertex, so we don't add it to the
            // positions vector
            positionIndices.push_back(positionToIndex[position]);
        }
        normals.emplace_back(idx.normal_index >= 0 ? glm::vec3(attributes.normals[3 * idx.normal_index + 0],
                                                               attributes.normals[3 * idx.normal_index + 1],
                                                               attributes.normals[3 * idx.normal_index + 2])
                                                   : glm::vec3(0.0f, 0.0f, 1.0f));
        uvs.emplace_back(idx.texcoord_index >= 0 ? glm::vec2(attributes.texcoords[2 * idx.texcoord_index + 0],
                                                             attributes.texcoords[2 * idx.texcoord_index + 1])
                                                 : glm::vec2(0.0f, 0.0f));

        return it->second;
    };

    for (const auto &shape : reader.GetShapes())
    {
        for (size_t faceIdx = 0; faceIdx < shape.mesh.num_face_vertices.size(); ++faceIdx)
        {
            // material_ids[faceIdx] is -1 when a face has no material even in a non-empty vector;
            // +1 in int space maps that (like an empty vector) to the default material at handle 0.
            faceGroups.emplace_back(shape.mesh.material_ids.empty() ? materialHandles.at(0)
                                                                    : materialHandles.at(static_cast<size_t>(
                                                                          shape.mesh.material_ids[faceIdx] + 1)));
            const size_t base = faceIdx * 3;
            faces.push_back({
                getOrAddVertex(shape.mesh.indices[base + 0]),
                getOrAddVertex(shape.mesh.indices[base + 1]),
                getOrAddVertex(shape.mesh.indices[base + 2]),
            });
        }
    }

    tangents.resize(positionIndices.size(), glm::vec4(0.0f));
    generateTangents(meshData);

    return meshData;
}

} // namespace

MeshLoadResult ObjLoader::load(const std::filesystem::path &path, MaterialStore &materialStore,
                               const ObjLoaderConfig &config)
{
    // SECTION 1 - Load the OBJ file

    if (path.empty())
    {
        throw std::invalid_argument("ObjLoader: empty path");
    }

    tinyobj::ObjReader reader = loadObjFile(path);

    // SECTION 2 - Extract material data and register it into the store first — mesh extraction
    // below needs the resulting handles to bake directly into each face's faceGroups entry.
    std::vector<MaterialHandle> materialHandles = extractMaterials(reader, path.parent_path(), config, materialStore);

    // SECTION 3 - Extract vertex / face data

    Mesh mesh;
    auto [positions, positionIndices, normals, tangents, uvs, faces, faceGroups] =
        extractMeshData(reader, path, materialHandles);

    mesh.setTopology(std::move(positions), std::move(positionIndices), std::move(faces));
    mesh.setFaceGroups(std::move(faceGroups));

    mesh.setPerVertexArray<glm::vec3>(config.normalAttributeName, normals);
    mesh.setPerVertexArray<glm::vec4>(config.tangentAttributeName, tangents);
    mesh.setPerVertexArray<glm::vec2>(config.uvAttributeName, uvs);

    MeshSequence sequence;
    sequence.frames.push_back(std::move(mesh));

    std::vector<MeshNode> nodes;
    nodes.push_back({.name = path.stem().string(),
                     .localTransform = Transform{},
                     .parent = std::nullopt,
                     .children = {},
                     .meshIndex = 0,
                     .skinIndex = std::nullopt});
    return {std::move(sequence), std::move(materialHandles), {}, std::move(nodes), {0}};
}

} // namespace lr
