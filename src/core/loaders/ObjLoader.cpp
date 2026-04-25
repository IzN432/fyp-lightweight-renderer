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

MaterialImage loadMaterialImage(const std::filesystem::path &objDirectoryPath,
                                const std::string &texName)
{
    if (texName.empty())
        return {};

    std::filesystem::path p(texName);
    if (!p.is_absolute())
        p = (objDirectoryPath / p).lexically_normal();

    if (!std::filesystem::exists(p))
        return {};

    LoadedImage loaded = loadImageFromFile(p);
    if (loaded.empty())
        return {};

    const size_t byteCount = static_cast<size_t>(loaded.width) * loaded.height * 4;
    MaterialImage out;
    out.width  = loaded.width;
    out.height = loaded.height;
    out.pixels.assign(loaded.pixels, loaded.pixels + byteCount);
    return out;
}

std::vector<Material> extractMaterials(const tinyobj::ObjReader &reader, const std::filesystem::path &objDirectoryPath, const ObjLoaderConfig &config)
{
    std::vector<Material> out;

    const auto &materials = reader.GetMaterials();
    out.reserve(materials.size() + 1);

    auto &defaultMat = out.emplace_back();
    defaultMat.name = "Default Material";
    defaultMat.parameters[config.baseDiffuseName] = glm::vec4(1.0f);
    defaultMat.parameters[config.baseAmbientName] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    defaultMat.parameters[config.baseSpecularName] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    defaultMat.parameters[config.shininessName] = 1.0f;
    defaultMat.parameters[config.baseRoughnessName] = 1.0f;
    defaultMat.parameters[config.baseMetallicName] = 0.0f;
    defaultMat.parameters[config.baseEmissiveName] = glm::vec3(0.0f);
    defaultMat.textures[config.diffuseTextureName] = MaterialImage::singlePixel(glm::vec4(1.0f));
    defaultMat.textures[config.ambientTextureName] = MaterialImage::singlePixel(glm::vec4(0.0f));
    defaultMat.textures[config.specularTextureName] = MaterialImage::singlePixel(glm::vec4(0.0f));
    defaultMat.textures[config.normalTextureName] = MaterialImage::singlePixel(glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
    defaultMat.textures[config.metallicTextureName] = MaterialImage::singlePixel(glm::vec4(0.0f));
    defaultMat.textures[config.roughnessTextureName] = MaterialImage::singlePixel(glm::vec4(1.0f));
    defaultMat.textures[config.emissiveTextureName] = MaterialImage::singlePixel(glm::vec4(0.0f));

    for (const auto &m : materials)
    {
        auto &mat = out.emplace_back();
        mat.name = m.name;
        mat.parameters[config.baseDiffuseName] = glm::vec4(m.diffuse[0], m.diffuse[1], m.diffuse[2], 1.0f);
        mat.parameters[config.baseAmbientName] = glm::vec4(m.ambient[0], m.ambient[1], m.ambient[2], 1.0f);
        mat.parameters[config.baseSpecularName] = glm::vec4(m.specular[0], m.specular[1], m.specular[2], 1.0f);
        mat.parameters[config.shininessName] = m.shininess;
        mat.parameters[config.baseRoughnessName] = m.roughness;
        mat.parameters[config.baseMetallicName] = m.metallic;
        mat.parameters[config.baseEmissiveName] = glm::vec3(m.emission[0], m.emission[1], m.emission[2]);
        mat.textures[config.diffuseTextureName] = loadMaterialImage(objDirectoryPath, m.diffuse_texname);
        mat.textures[config.ambientTextureName] = loadMaterialImage(objDirectoryPath, m.ambient_texname);
        mat.textures[config.specularTextureName] = loadMaterialImage(objDirectoryPath, m.specular_texname);
        mat.textures[config.normalTextureName] = loadMaterialImage(objDirectoryPath, m.normal_texname);
        mat.textures[config.metallicTextureName] = loadMaterialImage(objDirectoryPath, m.metallic_texname);
        mat.textures[config.roughnessTextureName] = loadMaterialImage(objDirectoryPath, m.roughness_texname); 
        mat.textures[config.emissiveTextureName] = loadMaterialImage(objDirectoryPath, m.emissive_texname);

        if (mat.textures[config.diffuseTextureName].pixels.empty())
            mat.textures[config.diffuseTextureName] = defaultMat.textures[config.diffuseTextureName];
        if (mat.textures[config.ambientTextureName].pixels.empty())
            mat.textures[config.ambientTextureName] = defaultMat.textures[config.ambientTextureName];
        if (mat.textures[config.specularTextureName].pixels.empty())    
            mat.textures[config.specularTextureName] = defaultMat.textures[config.specularTextureName];
        if (mat.textures[config.normalTextureName].pixels.empty())
            mat.textures[config.normalTextureName] = defaultMat.textures[config.normalTextureName];
        if (mat.textures[config.metallicTextureName].pixels.empty())
            mat.textures[config.metallicTextureName] = defaultMat.textures[config.metallicTextureName];
        if (mat.textures[config.roughnessTextureName].pixels.empty())
            mat.textures[config.roughnessTextureName] = defaultMat.textures[config.roughnessTextureName];
        if (mat.textures[config.emissiveTextureName].pixels.empty())
            mat.textures[config.emissiveTextureName] = defaultMat.textures[config.emissiveTextureName];
    }

    return out;
}

tinyobj::ObjReader loadObjFile(const std::filesystem::path &path)
{
    tinyobj::ObjReader reader;
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;

    if (!reader.ParseFromFile(path.string(), config))
    {
        std::string msg = "ObjLoader: failed to parse '" + path.string() + "'";
        if (!reader.Error().empty())
            msg += ": " + reader.Error();
        throw std::runtime_error(msg);
    }

    if (!reader.Warning().empty())
        spdlog::warn("ObjLoader warning while parsing '{}': {}", path.string(), reader.Warning());

    return reader;
}




struct VertexKey
{
    int positionIdx = -1;
    int uvIdx = -1;
    int normalIdx = -1;

    bool operator==(const VertexKey &other) const
    {
        return positionIdx == other.positionIdx &&
               uvIdx == other.uvIdx &&
               normalIdx == other.normalIdx;
    }
};

struct VertexKeyHash
{
    std::size_t operator()(const VertexKey &key) const
    {
        std::size_t seed = 0;
        auto combine = [&](int val) {
            std::size_t h = std::hash<int>{}(val);
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };

        combine(key.positionIdx);
        combine(key.uvIdx);
        combine(key.normalIdx);
        return seed;
    }
};

MeshData extractMeshData(const tinyobj::ObjReader &reader, const std::filesystem::path &path)
{
    MeshData meshData;
    auto &positions = meshData.positions;
    auto &normals = meshData.normals;
    auto &tangents = meshData.tangents;
    auto &uvs = meshData.uvs;
    auto &faces = meshData.faces;
    auto &faceGroups = meshData.faceGroups;

    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> vertexMap;

    const tinyobj::attrib_t &attributes = reader.GetAttrib();

    positions.reserve(attributes.vertices.size() / 3);
    normals.reserve(attributes.normals.size() / 3);
    uvs.reserve(attributes.texcoords.size() / 2);

    size_t indicesCount = 0;
    for (const auto &shape : reader.GetShapes())
        indicesCount += shape.mesh.indices.size();

    faces.reserve(indicesCount / 3);
    faceGroups.reserve(indicesCount / 3);

    auto getOrAddVertex = [&](const tinyobj::index_t &idx) -> uint32_t {
        VertexKey key{ idx.vertex_index, idx.texcoord_index, idx.normal_index };
        auto [it, inserted] = vertexMap.try_emplace(key, static_cast<uint32_t>(positions.size()));
        if (!inserted)
            return it->second;

        positions.emplace_back(
            attributes.vertices[3 * idx.vertex_index + 0],
            attributes.vertices[3 * idx.vertex_index + 1],
            attributes.vertices[3 * idx.vertex_index + 2]
        );
        normals.emplace_back(
            idx.normal_index >= 0
                ? glm::vec3(
                    attributes.normals[3 * idx.normal_index + 0],
                    attributes.normals[3 * idx.normal_index + 1],
                    attributes.normals[3 * idx.normal_index + 2])
                : glm::vec3(0.0f, 0.0f, 1.0f)
        );
        uvs.emplace_back(
            idx.texcoord_index >= 0
                ? glm::vec2(
                    attributes.texcoords[2 * idx.texcoord_index + 0],
                    attributes.texcoords[2 * idx.texcoord_index + 1])
                : glm::vec2(0.0f, 0.0f)
        );

        return it->second;
    };

    for (const auto &shape : reader.GetShapes())
    {
        for (size_t faceIdx = 0; faceIdx < shape.mesh.num_face_vertices.size(); ++faceIdx)
        {
            faceGroups.emplace_back(
                shape.mesh.material_ids.empty() ? 0 : (shape.mesh.material_ids[faceIdx] + 1)
            );
            const size_t base = faceIdx * 3;
            faces.push_back({
                getOrAddVertex(shape.mesh.indices[base + 0]),
                getOrAddVertex(shape.mesh.indices[base + 1]),
                getOrAddVertex(shape.mesh.indices[base + 2]),
            });
        }
    }

    tangents.resize(positions.size(), glm::vec4(0.0f));
    generateTangents(meshData);

    return meshData;
}

} // namespace

ObjMeshLoadResult ObjLoader::load(const std::filesystem::path &path, const ObjLoaderConfig &config) const
{
    // SECTION 1 - Load the OBJ file
    
    if (path.empty()) throw std::invalid_argument("ObjLoader: empty path");
    
    tinyobj::ObjReader reader = loadObjFile(path);

    // SECTION 2 - Extract vertex / face data

    MeshLayout layout;
    layout
        .addPerVertexAttr<glm::vec3>(config.normalAttributeName)
        .addPerVertexAttr<glm::vec4>(config.tangentAttributeName)
        .addPerVertexAttr<glm::vec2>(config.uvAttributeName);

    Mesh mesh(layout);
    auto [positions, normals, tangents, uvs, faces, faceGroups] = extractMeshData(reader, path);
    
    mesh.setVertexCount(static_cast<uint32_t>(positions.size()));
    mesh.setFaceCount(static_cast<uint32_t>(faces.size()));
    mesh.positions = std::move(positions);
    mesh.faces = std::move(faces);
    mesh.faceGroups = std::move(faceGroups);
    
    mesh.setPerVertexArray<glm::vec3>(config.normalAttributeName, normals);
    mesh.setPerVertexArray<glm::vec4>(config.tangentAttributeName, tangents);
    mesh.setPerVertexArray<glm::vec2>(config.uvAttributeName, uvs);

    // SECTION 3 - Extract material data from the tinyobj::ObjReader and convert it to our internal Material format
    auto materials = extractMaterials(reader, path.parent_path(), config);

    return { std::move(mesh), std::move(layout), std::move(materials) };
}

} // namespace lr