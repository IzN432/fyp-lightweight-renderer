#include "core/loaders/GltfLoader.hpp"

#include "glm/glm.hpp"
#include "core/loaders/LoaderUtils.hpp"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>
#include <spdlog/spdlog.h>

#include <optional>

namespace lr
{

namespace
{
// Load a glTF file into a tinygltf::Model, throwing on failure.
tinygltf::Model loadGltfFile(const std::filesystem::path &path)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    loader.SetPreserveImageChannels(false);

    const std::string pathStr = path.string();
    const std::string ext = path.extension().string();

    bool ok = (ext == ".glb")
                    ? loader.LoadBinaryFromFile(&model, &err, &warn, pathStr)
                    : loader.LoadASCIIFromFile(&model, &err, &warn, pathStr);

    if (!warn.empty())
        fprintf(stderr, "[WARN] %s\n", warn.c_str());
    if (!err.empty())
        throw std::runtime_error("GltfLoader: " + err);
    if (!ok)
        throw std::runtime_error("GltfLoader: failed to load model");

    return model;
}

glm::vec3 toVec3(const std::vector<double> &v)
{
    if (v.size() != 3)
        throw std::runtime_error("Expected a vec3");
    return glm::vec3(v[0], v[1], v[2]);
}

glm::vec4 toVec4(const std::vector<double> &v)
{
    if (v.size() != 4)
        throw std::runtime_error("Expected a vec4");
    return glm::vec4(v[0], v[1], v[2], v[3]);
}

/**
 * Extracts a MaterialImage from a tinygltf::TextureInfo, which references a tinygltf::Texture, which references a tinygltf::Image.
 * This ignores the sampler provided by the glTF file, which is a possible extension for future work.
 **/
MaterialImage extractImage(const tinygltf::Image &img)
{
    if (img.width <= 0 || img.height <= 0 || img.component <= 0)
        return {};

    MaterialImage materialImg;
    materialImg.name = img.name;
    materialImg.width = static_cast<uint32_t>(img.width);
    materialImg.height = static_cast<uint32_t>(img.height);
    // Assumes 4 channels, which is valid because we set loader.SetPreserveImageChannels(false)
    // in loadGltfFile(). If this becomes an issue we can add support for other channel counts in the future.
    materialImg.pixels = img.image;

    return materialImg;
}

MaterialImage extractImage(const tinygltf::TextureInfo &texInfo, const tinygltf::Model &model)
{
    if (texInfo.index < 0 || texInfo.index >= static_cast<int>(model.textures.size()))
        return {};

    const tinygltf::Texture &tex = model.textures[texInfo.index];
    if (tex.source < 0 || tex.source >= static_cast<int>(model.images.size()))
        return {};

    const tinygltf::Image &img = model.images[tex.source];
    return extractImage(img);
}

MaterialImage extractImage(const tinygltf::NormalTextureInfo &texInfo, const tinygltf::Model &model)
{
    if (texInfo.index < 0 || texInfo.index >= static_cast<int>(model.textures.size()))
        return {};

    const tinygltf::Texture &tex = model.textures[texInfo.index];
    if (tex.source < 0 || tex.source >= static_cast<int>(model.images.size()))
        return {};

    const tinygltf::Image &img = model.images[tex.source];
    return extractImage(img);
}

std::vector<Material> extractMaterials(const tinygltf::Model &model, const GltfLoaderConfig &config)
{
    std::vector<Material> materials;
    materials.reserve(model.materials.size() + 1);

    // Add a default material at index 0 for primitives that don't have a material
    Material &defaultMaterial = materials.emplace_back();
    defaultMaterial.name = "Default Material";
    defaultMaterial.parameters[config.baseDiffuseName] = glm::vec4(1.0f);
    defaultMaterial.parameters[config.baseRoughnessName] = 1.0f;
    defaultMaterial.parameters[config.baseMetallicName] = 0.0f;
    defaultMaterial.parameters[config.baseEmissiveName] = glm::vec3(0.0f);

    defaultMaterial.textures[config.diffuseTextureName] = MaterialImage::singlePixel(glm::vec4(1.0f));
    // Default normal texture points straight up. 0.5f is the "zero" value for normal maps, and the Z channel is usually stored in the B channel, so we set it to 1.0f.
    defaultMaterial.textures[config.normalTextureName] = MaterialImage::singlePixel(glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
    // Metallic is stored in the B channel and roughness is stored in the G channel, so we set metallic to 0.0f and roughness to 1.0f.
    defaultMaterial.textures[config.metallicRoughnessTextureName] = MaterialImage::singlePixel(glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
    defaultMaterial.textures[config.emissiveTextureName] = MaterialImage::singlePixel(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

    for (const auto &m : model.materials)
    {
        Material &material = materials.emplace_back();
        material.name = m.name;

        // The values in the Model are stored as doubles and vectors of doubles,
        // so we need to convert them to floats and glm::vec3/vec4.
        material.parameters[config.baseDiffuseName] = toVec4(m.pbrMetallicRoughness.baseColorFactor);
        material.parameters[config.baseRoughnessName] = static_cast<float>(m.pbrMetallicRoughness.roughnessFactor);
        material.parameters[config.baseMetallicName] = static_cast<float>(m.pbrMetallicRoughness.metallicFactor);
        material.parameters[config.baseEmissiveName] = toVec3(m.emissiveFactor);

        // The textures are stored in the material as tinygltf::TextureInfo, which contains a pointer to the actual texture
        material.textures[config.diffuseTextureName] = extractImage(m.pbrMetallicRoughness.baseColorTexture, model);
        // GLTF 2.0 uses a combined metallicRoughness texture, so we will store it in the metallicRoughnessTexture slot.
        // To be precise, it stores metallic in the B channel and roughness in the G channel, the shader must unpack it correctly.
        material.textures[config.normalTextureName] = extractImage(m.normalTexture, model);
        material.textures[config.metallicRoughnessTextureName] = extractImage(m.pbrMetallicRoughness.metallicRoughnessTexture, model);
        material.textures[config.emissiveTextureName] = extractImage(m.emissiveTexture, model);

        // Replace with the default if it is missing any of the required textures
        if (material.textures[config.diffuseTextureName].pixels.empty())
            material.textures[config.diffuseTextureName] = defaultMaterial.textures[config.diffuseTextureName];
        if (material.textures[config.normalTextureName].pixels.empty())
            material.textures[config.normalTextureName] = defaultMaterial.textures[config.normalTextureName];
        if (material.textures[config.metallicRoughnessTextureName].pixels.empty())
            material.textures[config.metallicRoughnessTextureName] = defaultMaterial.textures[config.metallicRoughnessTextureName];
        if (material.textures[config.emissiveTextureName].pixels.empty())
            material.textures[config.emissiveTextureName] = defaultMaterial.textures[config.emissiveTextureName];
    }
    return materials;
}

/**
 * Helper functions complementing extractMeshData
 */
float normalizeToFloat(const unsigned char *p, int componentType)
{
    switch (componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_BYTE:
        return std::max(-1.0f, static_cast<float>(*reinterpret_cast<const int8_t*>(p)) / 127.0f);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return static_cast<float>(*reinterpret_cast<const uint8_t*>(p)) / 255.0f;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
        return std::max(-1.0f, static_cast<float>(*reinterpret_cast<const int16_t*>(p)) / 32767.0f);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return static_cast<float>(*reinterpret_cast<const uint16_t*>(p)) / 65535.0f;
    default:
        // This is actually unreachable in a valid glTF for normalized attributes
        throw std::runtime_error("Invalid component type for normalized attribute");
    }
}

struct AccessorView
{
    const unsigned char *data = nullptr;
    size_t stride = 0;
    int type = -1;
    int componentType = -1;
    size_t numComponents = -1;
    bool normalized = false;
    size_t count = 0;
};

AccessorView getAccessorView(const tinygltf::Model &model, int accessorIndex)
{
    const tinygltf::Accessor &accessor = model.accessors[accessorIndex];
    const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

    AccessorView view;
    view.data = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
    view.stride = accessor.ByteStride(bufferView)
                    ? static_cast<size_t>(accessor.ByteStride(bufferView))
                    : tinygltf::GetComponentSizeInBytes(accessor.componentType) * tinygltf::GetNumComponentsInType(accessor.type);
    view.type = accessor.type;
    view.componentType = accessor.componentType;
    view.numComponents = tinygltf::GetNumComponentsInType(accessor.type);
    view.normalized = accessor.normalized;
    view.count = accessor.count;

    return view;
}

template <typename T>
T readVec(const AccessorView &view, size_t index)
{
    const unsigned char *data = view.data + index * view.stride;
    const size_t componentSize = tinygltf::GetComponentSizeInBytes(view.componentType);
    const int numComponents = T::length();

    T result;
    if (view.normalized)
    {
        for (int i = 0; i < numComponents; ++i)
            result[i] = normalizeToFloat(data + i * componentSize, view.componentType);
    }
    else
    {
        if (view.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
            throw std::runtime_error("Non-normalized attributes must be of type FLOAT");
        std::memcpy(&result, data, sizeof(T));
    }
    return result;
}

uint32_t readIndex(const AccessorView &view, size_t i)
{
    const unsigned char *data = view.data + i * view.stride;
    switch (view.componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  
        return *reinterpret_cast<const uint8_t *>(data);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: 
        return *reinterpret_cast<const uint16_t *>(data);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   
        return *reinterpret_cast<const uint32_t *>(data);
    default: 
        throw std::runtime_error("GltfLoader: unsupported index component type");
    }
}

static const std::string POSITION_ATTR_NAME = "POSITION";
static const std::string NORMAL_ATTR_NAME   = "NORMAL";
static const std::string UV_ATTR_NAME       = "TEXCOORD_0";
static const std::string TANGENT_ATTR_NAME  = "TANGENT";

MeshData extractMeshData(const tinygltf::Mesh &mesh, const tinygltf::Model &model)
{
    MeshData meshData;
    auto &positions = meshData.positions;
    auto &normals = meshData.normals;
    auto &tangents = meshData.tangents;
    auto &uvs = meshData.uvs;
    auto &faces = meshData.faces;
    auto &faceGroups = meshData.faceGroups;

    size_t totalVertexCount = 0;
    size_t totalFaceCount = 0;

    bool hasTangents = true;

    for (const auto &primitive : mesh.primitives)
    {
        AccessorView posView = getAccessorView(model, primitive.attributes.at(POSITION_ATTR_NAME));
        totalVertexCount += posView.count;
        totalFaceCount += primitive.indices >= 0 ? getAccessorView(model, primitive.indices).count / 3 : posView.count / 3;

        // We only use the provided tangents if they are present for every vertex. Otherwise, we will generate them ourselves
        if (primitive.attributes.find(TANGENT_ATTR_NAME) == primitive.attributes.end())
        {
            hasTangents = false;
            continue;
        }
        AccessorView tanView = getAccessorView(model, primitive.attributes.at(TANGENT_ATTR_NAME));
        if (tanView.count < posView.count)
        {
            hasTangents = false;
        }
    }

    positions.reserve(totalVertexCount);
    normals.reserve(totalVertexCount);
    tangents.reserve(totalVertexCount);
    uvs.reserve(totalVertexCount);
    faces.reserve(totalFaceCount);
    faceGroups.reserve(totalFaceCount);

    // For each primitive, which is a submesh, not a triangle
    for (const tinygltf::Primitive &primitive : mesh.primitives)
    {
        size_t baseVertex = positions.size();
        // We only support triangles for now, so we skip other primitive types
        if (primitive.mode != TINYGLTF_MODE_TRIANGLES)
        {
            spdlog::warn("GltfLoader: skipping primitive with unsupported mode {}", primitive.mode);
            continue;
        }

        // Extract the attributes

        // Tinygltf stores the attributes in accessors. The data is typically uploaded directly to the GPU
        // and accessed with the values in the accessor, but for our purposes, which involve modifying the
        // data in real-time, it is more convenient to have the data stored in an easy to understand format in CPU.

        // ATTRIBUTE 1 - Position (required)
        AccessorView posView = getAccessorView(model, primitive.attributes.at(POSITION_ATTR_NAME));
        size_t vertexCount = posView.count;

        if (posView.type != TINYGLTF_TYPE_VEC3)
        {
            spdlog::info(posView.type);
            throw std::runtime_error("GltfLoader: POSITION attribute must be of type VEC3");
        }
        if (posView.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
        {
            throw std::runtime_error("GltfLoader: POSITION attribute must be of component type FLOAT");
        }

        for (size_t j = 0; j < vertexCount; ++j)
        {
            positions.push_back(readVec<glm::vec3>(posView, j));
        }

        // ATTRIBUTE 2 - Normal
        
        if (primitive.attributes.find(NORMAL_ATTR_NAME) == primitive.attributes.end())
        {
            // If the normal accessor has fewer elements than the position accessor, we consider it as missing and fill with default value.
            spdlog::warn("GltfLoader: NORMAL attribute has fewer elements than POSITION attribute, filling with default value");
            for (size_t j = 0; j < vertexCount; ++j)
            {
                normals.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
            }
        }
        else
        {
            AccessorView normView = getAccessorView(model, primitive.attributes.at(NORMAL_ATTR_NAME));
            for (size_t j = 0; j < vertexCount; ++j)
            {
                normals.push_back(readVec<glm::vec3>(normView, j));
            }
        }

        // ATTRIBUTE 3 - UVs
        if (primitive.attributes.find(UV_ATTR_NAME) == primitive.attributes.end())
        {
            // If the UV accessor has fewer elements than the position accessor, we consider it as missing and fill with default value.
            spdlog::warn("GltfLoader: {} attribute has fewer elements than POSITION attribute, filling with default value", UV_ATTR_NAME);
            for (size_t j = 0; j < vertexCount; ++j)
            {
                uvs.push_back(glm::vec2(0.0f, 0.0f));
            }
        }
        else
        {
            AccessorView uvView = getAccessorView(model, primitive.attributes.at(UV_ATTR_NAME));
            
            for (size_t j = 0; j < vertexCount; ++j)
            {
                uvs.push_back(readVec<glm::vec2>(uvView, j));
            }
        }

         // ATTRIBUTE 4 - Tangent
        if (!hasTangents)
        {
            // Do nothing, we will calculate them later.
        }
        else
        {
            AccessorView tanView = getAccessorView(model, primitive.attributes.at(TANGENT_ATTR_NAME));
            
            for (size_t j = 0; j < vertexCount; ++j)
            {
                tangents.push_back(readVec<glm::vec4>(tanView, j));
            }
        }

        // Extract the faces of the primitive
        if (primitive.indices < 0)
        {
            // No indices, means the vertices were already in order.
            for (size_t j = 0; j < vertexCount; j += 3)
            {
                faces.push_back(glm::uvec3(baseVertex + j, baseVertex + j + 1, baseVertex + j + 2));
            }
        }
        else
        {
            AccessorView indexView = getAccessorView(model, primitive.indices);
            size_t faceCount = static_cast<size_t>(indexView.count) / 3;
            for (size_t j = 0; j < faceCount; ++j)
            {
                const uint32_t i0 = readIndex(indexView, j * 3 + 0);
                const uint32_t i1 = readIndex(indexView, j * 3 + 1);
                const uint32_t i2 = readIndex(indexView, j * 3 + 2);
                faces.push_back(glm::uvec3(baseVertex + i0, baseVertex + i1, baseVertex + i2));

                
                faceGroups.push_back(
                    primitive.material >= 0 
                    ? static_cast<uint32_t>(primitive.material) + 1
                    : 0);
            }
        }
    }

    // Generate tangents using mikktspace.

    if (!hasTangents)
    {
        tangents.resize(positions.size(), glm::vec4(0.0f));
        generateTangents(meshData);
    }

    return meshData;
}

} // namespace

GltfMeshLoadResult GltfLoader::load(const std::filesystem::path &path, const GltfLoaderConfig &config) const
{
    // SECTION 1 - Load the glTF file using tinygltf to obtain a tinygltf::Model instance.

    if (path.empty()) throw std::invalid_argument("GltfLoader: empty path");

    tinygltf::Model model = loadGltfFile(path);

    // SECTION 2 - Extract vertex / face data from the tinygltf::Model
    
    MeshLayout layout;
    layout
        .addPerVertexAttr<glm::vec3>(config.normalAttributeName)
        .addPerVertexAttr<glm::vec4>(config.tangentAttributeName)
        .addPerVertexAttr<glm::vec2>(config.uvAttributeName);
    
    MeshSequence seq;
    seq.frames.reserve(model.meshes.size());
    for (size_t i = 0; i < model.meshes.size(); ++i)
    {
        Mesh &outMesh = seq.frames.emplace_back(layout);
        const tinygltf::Mesh &mesh = model.meshes[i];
        
        if (mesh.primitives.empty())
            throw std::runtime_error("GltfLoader: mesh has no primitives");

        auto [positions, normals, tangents, uvs, faces, faceGroups] = extractMeshData(mesh, model);
        
        outMesh.setFaceCount(faces.size());
        outMesh.setVertexCount(positions.size());
        outMesh.positions = std::move(positions);
        outMesh.faces = std::move(faces);
        outMesh.faceGroups = std::move(faceGroups);

        outMesh.setPerVertexArray<glm::vec3>(config.normalAttributeName, normals);
        outMesh.setPerVertexArray<glm::vec4>(config.tangentAttributeName, tangents);
        outMesh.setPerVertexArray<glm::vec2>(config.uvAttributeName, uvs);
    }

    // SECTION 3 - Extract material data from the tinygltf::Model and convert it to our internal Material format.
    auto outMaterials = extractMaterials(model, config);

    return { std::move(seq), std::move(layout), std::move(outMaterials) };
}

} // namespace lr
