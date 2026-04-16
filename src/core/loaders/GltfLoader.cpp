#include "core/loaders/GltfLoader.hpp"

#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <unordered_set>
#include <vector>

namespace lr
{

namespace
{

std::filesystem::path resolveTexturePath(const std::filesystem::path &assetPath,
                                         const tinygltf::Model &model,
                                         int textureIndex)
{
    if (textureIndex < 0 || textureIndex >= static_cast<int>(model.textures.size()))
        return {};

    const tinygltf::Texture &tex = model.textures[textureIndex];
    if (tex.source < 0 || tex.source >= static_cast<int>(model.images.size()))
        return {};

    const tinygltf::Image &img = model.images[tex.source];
    if (img.uri.empty())
        return {}; // Embedded images (e.g. GLB) have no filesystem URI.

    std::filesystem::path p(img.uri);
    if (p.is_absolute())
        return p.lexically_normal();

    return (assetPath.parent_path() / p).lexically_normal();
}

std::vector<MaterialInfo> buildMaterialTable(const std::filesystem::path &assetPath,
                                             const tinygltf::Model &model)
{
    std::vector<MaterialInfo> out;
    out.reserve(model.materials.size());

    for (const tinygltf::Material &m : model.materials)
    {
        MaterialInfo info{};
        info.name = m.name;

        const auto &pbr = m.pbrMetallicRoughness;
        if (pbr.baseColorFactor.size() >= 3)
        {
            info.baseColorFactor = glm::vec3(
                static_cast<float>(pbr.baseColorFactor[0]),
                static_cast<float>(pbr.baseColorFactor[1]),
                static_cast<float>(pbr.baseColorFactor[2]));
        }

        info.roughnessFactor = static_cast<float>(pbr.roughnessFactor);
        info.metallicFactor  = static_cast<float>(pbr.metallicFactor);

        if (m.emissiveFactor.size() >= 3)
        {
            info.emissiveFactor = glm::vec3(
                static_cast<float>(m.emissiveFactor[0]),
                static_cast<float>(m.emissiveFactor[1]),
                static_cast<float>(m.emissiveFactor[2]));
        }

        info.textures.baseColor = resolveTexturePath(assetPath, model, pbr.baseColorTexture.index);
        info.textures.normal    = resolveTexturePath(assetPath, model, m.normalTexture.index);
        info.textures.emissive  = resolveTexturePath(assetPath, model, m.emissiveTexture.index);

        // Core glTF packs metallic + roughness into one texture by convention.
        const std::filesystem::path mr = resolveTexturePath(assetPath, model, pbr.metallicRoughnessTexture.index);
        info.textures.metallic  = mr;
        info.textures.roughness = mr;

        out.push_back(std::move(info));
    }

    return out;
}

std::vector<std::filesystem::path> gatherUniqueTextures(const std::vector<MaterialInfo> &materials)
{
    std::vector<std::filesystem::path> textures;
    std::unordered_set<std::string> seen;

    auto addIfValid = [&](const std::filesystem::path &p) {
        if (p.empty())
            return;
        const std::string key = p.generic_string();
        if (seen.insert(key).second)
            textures.push_back(p);
    };

    for (const MaterialInfo &m : materials)
    {
        addIfValid(m.textures.baseColor);
        addIfValid(m.textures.specular);
        addIfValid(m.textures.normal);
        addIfValid(m.textures.roughness);
        addIfValid(m.textures.metallic);
        addIfValid(m.textures.emissive);
    }

    return textures;
}

struct AccessorView
{
    const unsigned char *data = nullptr;
    size_t               stride = 0;
    int                  componentType = -1;
    int                  type = -1;
    bool                 normalized = false;
    size_t               count = 0;
};

size_t componentSize(int componentType)
{
    switch (componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
    case TINYGLTF_COMPONENT_TYPE_BYTE:
        return 1;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
    case TINYGLTF_COMPONENT_TYPE_SHORT:
        return 2;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
    case TINYGLTF_COMPONENT_TYPE_INT:
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        return 4;
    default:
        return 0;
    }
}

size_t componentCount(int type)
{
    switch (type)
    {
    case TINYGLTF_TYPE_SCALAR: return 1;
    case TINYGLTF_TYPE_VEC2:   return 2;
    case TINYGLTF_TYPE_VEC3:   return 3;
    case TINYGLTF_TYPE_VEC4:   return 4;
    default:                   return 0;
    }
}

AccessorView makeAccessorView(const tinygltf::Model &model, int accessorIndex)
{
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size()))
        throw std::runtime_error("GltfLoader: accessor index out of range");

    const tinygltf::Accessor &accessor = model.accessors[accessorIndex];
    if (accessor.bufferView < 0 || accessor.bufferView >= static_cast<int>(model.bufferViews.size()))
        throw std::runtime_error("GltfLoader: sparse/invalid accessor bufferView is not supported");

    const tinygltf::BufferView &view = model.bufferViews[accessor.bufferView];
    if (view.buffer < 0 || view.buffer >= static_cast<int>(model.buffers.size()))
        throw std::runtime_error("GltfLoader: accessor buffer index out of range");

    const tinygltf::Buffer &buffer = model.buffers[view.buffer];

    const size_t compSz = componentSize(accessor.componentType);
    const size_t compCt = componentCount(accessor.type);
    if (compSz == 0 || compCt == 0)
        throw std::runtime_error("GltfLoader: unsupported accessor component/type combination");

    const size_t packedStride = compSz * compCt;
    const size_t stride = accessor.ByteStride(view) ? static_cast<size_t>(accessor.ByteStride(view)) : packedStride;

    const size_t byteOffset = static_cast<size_t>(view.byteOffset) + static_cast<size_t>(accessor.byteOffset);
    if (byteOffset >= buffer.data.size())
        throw std::runtime_error("GltfLoader: accessor byte offset out of bounds");

    return AccessorView{
        .data = buffer.data.data() + byteOffset,
        .stride = stride,
        .componentType = accessor.componentType,
        .type = accessor.type,
        .normalized = accessor.normalized,
        .count = static_cast<size_t>(accessor.count),
    };
}

uint32_t readIndex(const AccessorView &v, size_t index)
{
    const unsigned char *p = v.data + index * v.stride;
    switch (v.componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return static_cast<uint32_t>(*reinterpret_cast<const uint8_t *>(p));
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(p));
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        return *reinterpret_cast<const uint32_t *>(p);
    default:
        throw std::runtime_error("GltfLoader: unsupported index accessor component type");
    }
}

float normalizedToFloat(const unsigned char *p, int componentType)
{
    switch (componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return static_cast<float>(*reinterpret_cast<const uint8_t *>(p)) / 255.0f;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return static_cast<float>(*reinterpret_cast<const uint16_t *>(p)) / 65535.0f;
    default:
        throw std::runtime_error("GltfLoader: unsupported normalized component type");
    }
}

glm::vec3 readVec3(const AccessorView &v, size_t index)
{
    if (v.type != TINYGLTF_TYPE_VEC3 || v.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
        throw std::runtime_error("GltfLoader: expected VEC3 FLOAT accessor");

    const float *f = reinterpret_cast<const float *>(v.data + index * v.stride);
    return glm::vec3(f[0], f[1], f[2]);
}

glm::vec2 readVec2(const AccessorView &v, size_t index)
{
    const unsigned char *p = v.data + index * v.stride;
    if (v.type != TINYGLTF_TYPE_VEC2)
        throw std::runtime_error("GltfLoader: expected VEC2 accessor for TEXCOORD_0");

    if (v.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
    {
        const float *f = reinterpret_cast<const float *>(p);
        return glm::vec2(f[0], f[1]);
    }

    if (v.normalized &&
        (v.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE ||
         v.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT))
    {
        const size_t cs = componentSize(v.componentType);
        return glm::vec2(normalizedToFloat(p + 0 * cs, v.componentType),
                         normalizedToFloat(p + 1 * cs, v.componentType));
    }

    throw std::runtime_error("GltfLoader: unsupported TEXCOORD_0 accessor format");
}

Mesh loadGltfMesh(const tinygltf::Model &model, const MeshLayout &layout)
{
    const auto *normalDesc = layout.findPerVertexAttr("normal");
    const auto *uvDesc = layout.findPerVertexAttr("uv");
    const auto *texcoordDesc = layout.findPerVertexAttr("texcoord");
    const auto *materialIdDesc = layout.findPerFaceAttr("materialId");
    const auto *faceGroupMaterialIdDesc = layout.findFaceGroupAttr("materialId");

    const bool hasNormalAttr = (normalDesc && normalDesc->type == std::type_index(typeid(glm::vec3)));
    const bool hasUvAttr =
        (uvDesc && uvDesc->type == std::type_index(typeid(glm::vec2))) ||
        (texcoordDesc && texcoordDesc->type == std::type_index(typeid(glm::vec2)));
    const bool hasMaterialIdAttr = materialIdDesc &&
        (materialIdDesc->type == std::type_index(typeid(int32_t)) ||
         materialIdDesc->type == std::type_index(typeid(uint32_t)));
    const bool hasFaceGroupMaterialIdAttr = faceGroupMaterialIdDesc &&
        (faceGroupMaterialIdDesc->type == std::type_index(typeid(int32_t)) ||
         faceGroupMaterialIdDesc->type == std::type_index(typeid(uint32_t)));
    const bool needsMaterialIds = hasMaterialIdAttr || layout.faceGroupsEnabled() || hasFaceGroupMaterialIdAttr;

    const std::string uvAttrName = (uvDesc && uvDesc->type == std::type_index(typeid(glm::vec2)))
        ? "uv"
        : "texcoord";

    std::vector<glm::vec3> positions;
    std::vector<glm::uvec3> faces;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<int32_t> materialIds;

    for (const tinygltf::Mesh &mesh : model.meshes)
    {
        for (const tinygltf::Primitive &prim : mesh.primitives)
        {
            const int mode = (prim.mode == -1) ? TINYGLTF_MODE_TRIANGLES : prim.mode;
            if (mode != TINYGLTF_MODE_TRIANGLES)
                throw std::runtime_error("GltfLoader: only TRIANGLES primitives are supported");

            auto posIt = prim.attributes.find("POSITION");
            if (posIt == prim.attributes.end())
                throw std::runtime_error("GltfLoader: primitive is missing POSITION accessor");

            AccessorView posView = makeAccessorView(model, posIt->second);
            if (posView.type != TINYGLTF_TYPE_VEC3 || posView.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
                throw std::runtime_error("GltfLoader: POSITION must be VEC3 FLOAT");

            bool hasNormalData = false;
            AccessorView normalView{};
            if (hasNormalAttr)
            {
                auto nIt = prim.attributes.find("NORMAL");
                if (nIt != prim.attributes.end())
                {
                    normalView = makeAccessorView(model, nIt->second);
                    if (normalView.count != posView.count)
                        throw std::runtime_error("GltfLoader: NORMAL count does not match POSITION count");
                    hasNormalData = true;
                }
            }

            bool hasUvData = false;
            AccessorView uvView{};
            if (hasUvAttr)
            {
                auto uvIt = prim.attributes.find("TEXCOORD_0");
                if (uvIt != prim.attributes.end())
                {
                    uvView = makeAccessorView(model, uvIt->second);
                    if (uvView.count != posView.count)
                        throw std::runtime_error("GltfLoader: TEXCOORD_0 count does not match POSITION count");
                    hasUvData = true;
                }
            }

            const uint32_t baseVertex = static_cast<uint32_t>(positions.size());
            positions.reserve(positions.size() + posView.count);
            if (hasNormalAttr)
                normals.reserve(normals.size() + posView.count);
            if (hasUvAttr)
                uvs.reserve(uvs.size() + posView.count);

            for (size_t i = 0; i < posView.count; ++i)
            {
                positions.push_back(readVec3(posView, i));

                if (hasNormalAttr)
                {
                    const glm::vec3 n = hasNormalData ? readVec3(normalView, i) : glm::vec3(0.0f);
                    normals.push_back(n);
                }

                if (hasUvAttr)
                {
                    const glm::vec2 uv = hasUvData ? readVec2(uvView, i) : glm::vec2(0.0f);
                    uvs.push_back(uv);
                }
            }

            uint32_t primitiveFaceCount = 0;
            if (prim.indices >= 0)
            {
                AccessorView idxView = makeAccessorView(model, prim.indices);
                if (idxView.type != TINYGLTF_TYPE_SCALAR)
                    throw std::runtime_error("GltfLoader: index accessor must be SCALAR");
                if (idxView.count % 3 != 0)
                    throw std::runtime_error("GltfLoader: index count is not divisible by 3");

                primitiveFaceCount = static_cast<uint32_t>(idxView.count / 3);
                faces.reserve(faces.size() + primitiveFaceCount);
                for (size_t i = 0; i < idxView.count; i += 3)
                {
                    const uint32_t i0 = readIndex(idxView, i + 0);
                    const uint32_t i1 = readIndex(idxView, i + 1);
                    const uint32_t i2 = readIndex(idxView, i + 2);
                    faces.push_back(glm::uvec3(baseVertex + i0, baseVertex + i1, baseVertex + i2));
                }
            }
            else
            {
                if (posView.count % 3 != 0)
                    throw std::runtime_error("GltfLoader: non-indexed primitive vertex count is not divisible by 3");

                primitiveFaceCount = static_cast<uint32_t>(posView.count / 3);
                faces.reserve(faces.size() + primitiveFaceCount);
                for (uint32_t i = 0; i < primitiveFaceCount; ++i)
                {
                    const uint32_t i0 = baseVertex + i * 3 + 0;
                    const uint32_t i1 = baseVertex + i * 3 + 1;
                    const uint32_t i2 = baseVertex + i * 3 + 2;
                    faces.push_back(glm::uvec3(i0, i1, i2));
                }
            }

            if (needsMaterialIds)
            {
                int32_t mat = prim.material;
                if (mat < 0 || mat >= static_cast<int32_t>(model.materials.size()))
                    mat = -1;
                materialIds.insert(materialIds.end(), primitiveFaceCount, mat);
            }
        }
    }

    if (positions.empty() || faces.empty())
        throw std::runtime_error("GltfLoader: no triangle geometry found");

    Mesh mesh(layout);
    mesh.setVertexCount(static_cast<uint32_t>(positions.size()));
    mesh.setFaceCount(static_cast<uint32_t>(faces.size()));
    mesh.positions = std::move(positions);
    mesh.faces = std::move(faces);

    if (hasNormalAttr)
        mesh.setPerVertexArray<glm::vec3>("normal", normals);

    if (hasUvAttr)
        mesh.setPerVertexArray<glm::vec2>(uvAttrName, uvs);

    if (hasMaterialIdAttr)
    {
        if (materialIdDesc->type == std::type_index(typeid(int32_t)))
        {
            mesh.setPerFaceArray<int32_t>("materialId", materialIds);
        }
        else
        {
            std::vector<uint32_t> mats;
            mats.reserve(materialIds.size());
            for (int32_t m : materialIds)
                mats.push_back(m < 0 ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(m));
            mesh.setPerFaceArray<uint32_t>("materialId", mats);
        }
    }

    if (layout.faceGroupsEnabled())
    {
        bool hasInvalidMaterial = false;
        for (int32_t materialId : materialIds)
        {
            if (materialId < 0)
            {
                hasInvalidMaterial = true;
                break;
            }
        }

        const uint32_t validGroupCount = static_cast<uint32_t>(model.materials.size());
        const uint32_t invalidGroupIndex = validGroupCount;
        const uint32_t faceGroupCount = hasInvalidMaterial
            ? invalidGroupIndex + 1u
            : (validGroupCount > 0u ? validGroupCount : 1u);

        mesh.setFaceGroupCount(faceGroupCount);
        mesh.faceGroups.resize(mesh.faceCount(), invalidGroupIndex);
        for (uint32_t faceIndex = 0; faceIndex < mesh.faceCount(); ++faceIndex)
        {
            const int32_t materialId = (faceIndex < materialIds.size()) ? materialIds[faceIndex] : -1;
            mesh.faceGroups[faceIndex] = (materialId < 0)
                ? invalidGroupIndex
                : static_cast<uint32_t>(materialId);
        }

        if (hasFaceGroupMaterialIdAttr)
        {
            if (faceGroupMaterialIdDesc->type == std::type_index(typeid(int32_t)))
            {
                std::vector<int32_t> groupMaterialIds(faceGroupCount, -1);
                for (uint32_t materialIndex = 0; materialIndex < model.materials.size(); ++materialIndex)
                    groupMaterialIds[materialIndex] = static_cast<int32_t>(materialIndex);
                mesh.setFaceGroupAttributeArray<int32_t>("materialId", groupMaterialIds);
            }
            else
            {
                std::vector<uint32_t> groupMaterialIds(faceGroupCount, std::numeric_limits<uint32_t>::max());
                for (uint32_t materialIndex = 0; materialIndex < model.materials.size(); ++materialIndex)
                    groupMaterialIds[materialIndex] = materialIndex;
                mesh.setFaceGroupAttributeArray<uint32_t>("materialId", groupMaterialIds);
            }
        }
    }

    return mesh;
}

} // namespace

MeshSequence GltfLoader::load(const std::filesystem::path &path,
                              const MeshLayout &layout) const
{
    if (path.empty())
        throw std::invalid_argument("GltfLoader: empty mesh path");

    const std::string ext = path.extension().string();
    if (ext != ".gltf" && ext != ".glb" && ext != ".GLTF" && ext != ".GLB")
    {
        throw std::runtime_error(
            "GltfLoader: unsupported extension '" + ext + "' for file '" + path.string() + "'");
    }

    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err;
    std::string warn;

    const bool isBinary = (ext == ".glb" || ext == ".GLB");
    const bool ok = isBinary
        ? loader.LoadBinaryFromFile(&model, &err, &warn, path.string())
        : loader.LoadASCIIFromFile(&model, &err, &warn, path.string());

    if (!warn.empty())
        spdlog::warn("GltfLoader: {}", warn);

    if (!ok)
    {
        std::string msg = "GltfLoader: failed to parse '" + path.string() + "'";
        if (!err.empty())
            msg += ": " + err;
        throw std::runtime_error(msg);
    }

    MeshSequence seq;
    seq.frames.push_back(loadGltfMesh(model, layout));
    seq.frameRate = 0.0f;
    seq.materials = buildMaterialTable(path, model);
    seq.textureFiles = gatherUniqueTextures(seq.materials);
    return seq;
}

}  // namespace lr
