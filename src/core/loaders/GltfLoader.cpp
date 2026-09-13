#include "core/loaders/GltfLoader.hpp"

#include "glm/glm.hpp"
#include "core/loaders/LoaderUtils.hpp"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_map>

namespace lr
{

namespace
{
// Load a glTF file into a tinygltf::Model, throwing on failure.
tinygltf::Model loadGltfFile(const std::filesystem::path &path)
{
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string        err, warn;

    loader.SetPreserveImageChannels(false);

    const std::string pathStr = path.string();
    const std::string ext     = path.extension().string();

    bool ok = (ext == ".glb") ? loader.LoadBinaryFromFile(&model, &err, &warn, pathStr)
                              : loader.LoadASCIIFromFile(&model, &err, &warn, pathStr);

    if (!warn.empty())
    {
        fprintf(stderr, "[WARN] %s\n", warn.c_str());
    }
    if (!err.empty())
    {
        throw std::runtime_error("GltfLoader: " + err);
    }
    if (!ok)
    {
        throw std::runtime_error("GltfLoader: failed to load model");
    }

    return model;
}

glm::vec3 toVec3(const std::vector<double> &v)
{
    if (v.size() != 3)
    {
        throw std::runtime_error("Expected a vec3");
    }
    return glm::vec3(v[0], v[1], v[2]);
}

glm::vec4 toVec4(const std::vector<double> &v)
{
    if (v.size() != 4)
    {
        throw std::runtime_error("Expected a vec4");
    }
    return glm::vec4(v[0], v[1], v[2], v[3]);
}

Transform extractNodeTransform(const tinygltf::Node &node)
{
    if (!node.matrix.empty())
    {
        if (node.matrix.size() != 16)
        {
            throw std::runtime_error("GltfLoader: node matrix must contain 16 values");
        }

        glm::mat4 matrix(1.0f);
        for (size_t column = 0; column < 4; ++column)
        {
            for (size_t row = 0; row < 4; ++row)
            {
                matrix[column][row] = static_cast<float>(node.matrix[column * 4 + row]);
            }
        }

        constexpr float epsilon = 1e-5f;
        if (std::abs(matrix[0][3]) > epsilon || std::abs(matrix[1][3]) > epsilon ||
            std::abs(matrix[2][3]) > epsilon || std::abs(matrix[3][3] - 1.0f) > epsilon)
        {
            throw std::runtime_error("GltfLoader: node matrix contains perspective and cannot be represented by Transform");
        }

        const glm::vec3 translation(matrix[3]);
        glm::vec3       xAxis(matrix[0]);
        glm::vec3       yAxis(matrix[1]);
        glm::vec3       zAxis(matrix[2]);
        glm::vec3       scale(glm::length(xAxis), glm::length(yAxis), glm::length(zAxis));
        if (scale.x <= epsilon || scale.y <= epsilon || scale.z <= epsilon)
        {
            throw std::runtime_error("GltfLoader: node matrix with zero scale cannot be decomposed into Transform");
        }

        xAxis /= scale.x;
        yAxis /= scale.y;
        zAxis /= scale.z;
        if (std::abs(glm::dot(xAxis, yAxis)) > epsilon || std::abs(glm::dot(xAxis, zAxis)) > epsilon ||
            std::abs(glm::dot(yAxis, zAxis)) > epsilon)
        {
            throw std::runtime_error("GltfLoader: node matrix contains shear and cannot be represented by Transform");
        }

        glm::mat3 rotationMatrix(xAxis, yAxis, zAxis);
        if (glm::determinant(rotationMatrix) < 0.0f)
        {
            scale.x = -scale.x;
            rotationMatrix[0] = -rotationMatrix[0];
        }

        const glm::quat rotation = glm::normalize(glm::quat_cast(rotationMatrix));
        return Transform(translation, rotation, scale);
    }

    glm::vec3 translation(0.0f);
    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale(1.0f);

    if (!node.translation.empty())
    {
        translation = toVec3(node.translation);
    }
    if (!node.rotation.empty())
    {
        const glm::vec4 value = toVec4(node.rotation);
        rotation = glm::quat(value.w, value.x, value.y, value.z);
        const float length = glm::length(rotation);
        if (!std::isfinite(length) || length <= std::numeric_limits<float>::epsilon())
        {
            throw std::runtime_error("GltfLoader: node rotation must be a finite non-zero quaternion");
        }
        rotation = glm::normalize(rotation);
    }
    if (!node.scale.empty())
    {
        scale = toVec3(node.scale);
    }

    return Transform(translation, rotation, scale);
}

/**
 * Extracts a MaterialImage from a tinygltf::TextureInfo, which references a tinygltf::Texture, which references a
 * tinygltf::Image. This ignores the sampler provided by the glTF file, which is a possible extension for future work.
 **/
MaterialImage extractImage(const tinygltf::Image &img)
{
    if (img.width <= 0 || img.height <= 0 || img.component <= 0)
    {
        return {};
    }

    MaterialImage materialImg;
    materialImg.name   = img.name;
    materialImg.width  = static_cast<uint32_t>(img.width);
    materialImg.height = static_cast<uint32_t>(img.height);

    if (img.bits == 16)
    {
        // 16-bit-per-channel PNG: img.image holds uint16_t values as raw bytes.
        // Downscale to 8-bit by taking the high byte of each channel value.
        const size_t channelCount = static_cast<size_t>(img.width) * img.height * img.component;
        const auto  *src16        = reinterpret_cast<const uint16_t *>(img.image.data());
        materialImg.pixels.resize(channelCount);
        for (size_t i = 0; i < channelCount; ++i)
        {
            materialImg.pixels[i] = static_cast<uint8_t>(src16[i] >> 8);
        }
    } else
    {
        materialImg.pixels = img.image;
    }

    return materialImg;
}

MaterialImage extractImage(const tinygltf::TextureInfo &texInfo, const tinygltf::Model &model)
{
    if (texInfo.index < 0 || texInfo.index >= static_cast<int>(model.textures.size()))
    {
        return {};
    }

    const tinygltf::Texture &tex = model.textures[texInfo.index];
    if (tex.source < 0 || tex.source >= static_cast<int>(model.images.size()))
    {
        return {};
    }

    const tinygltf::Image &img = model.images[tex.source];
    return extractImage(img);
}

MaterialImage extractImage(const tinygltf::NormalTextureInfo &texInfo, const tinygltf::Model &model)
{
    if (texInfo.index < 0 || texInfo.index >= static_cast<int>(model.textures.size()))
    {
        return {};
    }

    const tinygltf::Texture &tex = model.textures[texInfo.index];
    if (tex.source < 0 || tex.source >= static_cast<int>(model.images.size()))
    {
        return {};
    }

    const tinygltf::Image &img = model.images[tex.source];
    return extractImage(img);
}

// Registers each material into `materialStore` as it's built and returns the resulting handles,
// parallel to the glTF material indices (index 0 = MaterialStore's shared default material, index
// i+1 = model.materials[i]) — extractMeshData bakes these straight into each face's faceGroups entry.
std::vector<MaterialHandle> extractMaterials(const tinygltf::Model &model, const GltfLoaderConfig &config,
                                             MaterialStore &materialStore)
{
    std::vector<MaterialHandle> handles;
    handles.reserve(model.materials.size() + 1);
    handles.push_back(materialStore.defaultMaterialHandle());

    // Fallback texture values used to patch any real material missing a required texture — just a
    // handful of 1x1 pixels, cheap to keep local rather than routing through the store.
    const MaterialImage fallbackDiffuse = MaterialImage::singlePixel(glm::vec4(1.0f));
    // Default normal texture points straight up. 0.5f is the "zero" value for normal maps, and the Z channel is usually
    // stored in the B channel, so we set it to 1.0f.
    const MaterialImage fallbackNormal = MaterialImage::singlePixel(glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
    // Metallic is stored in the B channel and roughness is stored in the G channel, so we set metallic to 0.0f and
    // roughness to 1.0f.
    const MaterialImage fallbackMetallicRoughness = MaterialImage::singlePixel(glm::vec4(0.0f, 1.0f, 1.0f, 1.0f));
    const MaterialImage fallbackEmissive          = MaterialImage::singlePixel(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

    for (const auto &m : model.materials)
    {
        Material material;
        material.name = m.name;

        // The values in the Model are stored as doubles and vectors of doubles,
        // so we need to convert them to floats and glm::vec3/vec4.
        material.parameters[config.baseDiffuseName] =
            MaterialParam::ColorRGBA{toVec4(m.pbrMetallicRoughness.baseColorFactor)};
        material.parameters[config.baseRoughnessName] =
            MaterialParam::RangedFloat{static_cast<float>(m.pbrMetallicRoughness.roughnessFactor), 0.0f, 1.0f};
        material.parameters[config.baseMetallicName] =
            MaterialParam::NormalizedFloat{static_cast<float>(m.pbrMetallicRoughness.metallicFactor)};
        material.parameters[config.baseEmissiveName] = MaterialParam::ColorRGB{toVec3(m.emissiveFactor)};

        // The textures are stored in the material as tinygltf::TextureInfo, which contains a pointer to the actual
        // texture
        material.textures[config.diffuseTextureName] = extractImage(m.pbrMetallicRoughness.baseColorTexture, model);
        // GLTF 2.0 uses a combined metallicRoughness texture, so we will store it in the metallicRoughnessTexture slot.
        // To be precise, it stores metallic in the B channel and roughness in the G channel, the shader must unpack it
        // correctly.
        material.textures[config.normalTextureName] = extractImage(m.normalTexture, model);
        material.textures[config.metallicRoughnessTextureName] =
            extractImage(m.pbrMetallicRoughness.metallicRoughnessTexture, model);
        material.textures[config.emissiveTextureName] = extractImage(m.emissiveTexture, model);

        // Replace with the default if it is missing any of the required textures
        if (material.textures[config.diffuseTextureName].pixels.empty())
        {
            material.textures[config.diffuseTextureName] = fallbackDiffuse;
        }
        if (material.textures[config.normalTextureName].pixels.empty())
        {
            material.textures[config.normalTextureName] = fallbackNormal;
        }
        if (material.textures[config.metallicRoughnessTextureName].pixels.empty())
        {
            material.textures[config.metallicRoughnessTextureName] = fallbackMetallicRoughness;
        }
        if (material.textures[config.emissiveTextureName].pixels.empty())
        {
            material.textures[config.emissiveTextureName] = fallbackEmissive;
        }

        handles.push_back(materialStore.acquire(std::move(material)));
    }
    return handles;
}

/**
 * Helper functions complementing extractMeshData
 */
float normalizeToFloat(const unsigned char *p, int componentType)
{
    switch (componentType)
    {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
            return std::max(-1.0f, static_cast<float>(*reinterpret_cast<const int8_t *>(p)) / 127.0f);
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return static_cast<float>(*reinterpret_cast<const uint8_t *>(p)) / 255.0f;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
            return std::max(-1.0f, static_cast<float>(*reinterpret_cast<const int16_t *>(p)) / 32767.0f);
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return static_cast<float>(*reinterpret_cast<const uint16_t *>(p)) / 65535.0f;
        default:
            // This is actually unreachable in a valid glTF for normalized attributes
            throw std::runtime_error("Invalid component type for normalized attribute");
    }
}

struct AccessorView
{
    const unsigned char *data          = nullptr;
    size_t               stride        = 0;
    int                  type          = -1;
    int                  componentType = -1;
    size_t               numComponents = -1;
    bool                 normalized    = false;
    size_t               count         = 0;
};

AccessorView getAccessorView(const tinygltf::Model &model, int accessorIndex)
{
    const tinygltf::Accessor   &accessor   = model.accessors[accessorIndex];
    const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer     &buffer     = model.buffers[bufferView.buffer];

    AccessorView view;
    view.data          = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
    view.stride        = accessor.ByteStride(bufferView) ? static_cast<size_t>(accessor.ByteStride(bufferView))
                                                         : tinygltf::GetComponentSizeInBytes(accessor.componentType) *
                                                        tinygltf::GetNumComponentsInType(accessor.type);
    view.type          = accessor.type;
    view.componentType = accessor.componentType;
    view.numComponents = tinygltf::GetNumComponentsInType(accessor.type);
    view.normalized    = accessor.normalized;
    view.count         = accessor.count;

    return view;
}

template <typename T> T readVec(const AccessorView &view, size_t index)
{
    const unsigned char *data          = view.data + index * view.stride;
    const size_t         componentSize = tinygltf::GetComponentSizeInBytes(view.componentType);
    const int            numComponents = T::length();

    T result;
    if (view.normalized)
    {
        for (int i = 0; i < numComponents; ++i)
        {
            result[i] = normalizeToFloat(data + i * componentSize, view.componentType);
        }
    } else
    {
        if (view.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
        {
            throw std::runtime_error("Non-normalized attributes must be of type FLOAT");
        }
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

void validateJointView(const AccessorView &view)
{
    if (view.type != TINYGLTF_TYPE_VEC4 || view.normalized ||
        (view.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE &&
         view.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT))
    {
        throw std::runtime_error(
            "GltfLoader: JOINTS_n must be a non-normalized VEC4 of unsigned bytes or unsigned shorts");
    }
}

glm::mat4 readMatrix(const AccessorView &view, size_t index)
{
    if (view.type != TINYGLTF_TYPE_MAT4 || view.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT || view.normalized)
    {
        throw std::runtime_error("GltfLoader: inverse bind matrices must be non-normalized FLOAT MAT4 values");
    }

    glm::mat4 result(1.0f);
    const unsigned char *data = view.data + index * view.stride;
    std::memcpy(&result[0][0], data, sizeof(glm::mat4));
    return result;
}

std::vector<std::optional<uint32_t>> buildNodeParents(const tinygltf::Model &model)
{
    std::vector<std::optional<uint32_t>> parents(model.nodes.size());
    for (uint32_t parentIndex = 0; parentIndex < model.nodes.size(); ++parentIndex)
    {
        for (int childIndex : model.nodes[parentIndex].children)
        {
            if (childIndex < 0 || childIndex >= static_cast<int>(model.nodes.size()))
            {
                throw std::runtime_error("GltfLoader: node references an invalid child");
            }

            auto &parent = parents[static_cast<size_t>(childIndex)];
            if (parent && parent.value() != parentIndex)
            {
                throw std::runtime_error("GltfLoader: a node cannot have multiple parents");
            }
            parent = parentIndex;
        }
    }
    return parents;
}

std::vector<Skin> extractSkins(const tinygltf::Model &model,
                               const std::vector<std::optional<uint32_t>> &gltfNodeParents)
{
    std::vector<Skin> skins;
    skins.reserve(model.skins.size());

    for (const tinygltf::Skin &sourceSkin : model.skins)
    {
        std::vector<bool> included(model.nodes.size(), false);
        for (int sourceJointIndex : sourceSkin.joints)
        {
            if (sourceJointIndex < 0 || sourceJointIndex >= static_cast<int>(model.nodes.size()))
            {
                throw std::runtime_error("GltfLoader: skin references an invalid joint node");
            }

            uint32_t current = static_cast<uint32_t>(sourceJointIndex);
            size_t   remaining = model.nodes.size();
            while (true)
            {
                included[current] = true;
                if (!gltfNodeParents[current])
                {
                    break;
                }
                if (remaining-- == 0)
                {
                    throw std::runtime_error("GltfLoader: node hierarchy contains a cycle");
                }
                current = gltfNodeParents[current].value();
            }
        }

        std::vector<std::optional<SkeletonNodeIndex>> sourceToLocal(model.nodes.size());
        std::vector<SkeletonNode>                     nodes;
        for (uint32_t sourceNodeIndex = 0; sourceNodeIndex < model.nodes.size(); ++sourceNodeIndex)
        {
            if (included[sourceNodeIndex])
            {
                sourceToLocal[sourceNodeIndex] = static_cast<SkeletonNodeIndex>(nodes.size());
                nodes.push_back({.name = model.nodes[sourceNodeIndex].name,
                                 .sourceNodeIndex = sourceNodeIndex,
                                 .parent = std::nullopt,
                                 .localTransform = extractNodeTransform(model.nodes[sourceNodeIndex])});
            }
        }

        for (SkeletonNode &node : nodes)
        {
            const auto sourceParent = gltfNodeParents[node.sourceNodeIndex];
            if (sourceParent)
            {
                node.parent = sourceToLocal[sourceParent.value()].value();
            }
        }

        std::vector<glm::mat4> inverseBindMatrices(sourceSkin.joints.size(), glm::mat4(1.0f));
        if (sourceSkin.inverseBindMatrices >= 0)
        {
            if (sourceSkin.inverseBindMatrices >= static_cast<int>(model.accessors.size()))
            {
                throw std::runtime_error("GltfLoader: skin references an invalid inverse-bind accessor");
            }
            const tinygltf::Accessor &accessor = model.accessors[sourceSkin.inverseBindMatrices];
            if (accessor.sparse.isSparse)
            {
                throw std::runtime_error("GltfLoader: sparse inverse-bind matrix accessors are not supported");
            }
            if (accessor.count != sourceSkin.joints.size())
            {
                throw std::runtime_error("GltfLoader: inverse-bind matrix count must match the joint count");
            }

            const AccessorView view = getAccessorView(model, sourceSkin.inverseBindMatrices);
            for (size_t jointIndex = 0; jointIndex < sourceSkin.joints.size(); ++jointIndex)
            {
                inverseBindMatrices[jointIndex] = readMatrix(view, jointIndex);
            }
        }

        std::vector<Joint> joints;
        joints.reserve(sourceSkin.joints.size());
        for (size_t jointIndex = 0; jointIndex < sourceSkin.joints.size(); ++jointIndex)
        {
            const uint32_t sourceNodeIndex = static_cast<uint32_t>(sourceSkin.joints[jointIndex]);
            joints.push_back({.node = sourceToLocal[sourceNodeIndex].value(),
                              .inverseBindMatrix = inverseBindMatrices[jointIndex]});
        }

        skins.emplace_back(std::move(nodes), std::move(joints));
    }

    return skins;
}

std::vector<GltfMeshInstance> extractMeshInstances(const tinygltf::Model &model)
{
    std::vector<GltfMeshInstance> instances;
    for (uint32_t nodeIndex = 0; nodeIndex < model.nodes.size(); ++nodeIndex)
    {
        const tinygltf::Node &node = model.nodes[nodeIndex];
        if (node.mesh < 0)
        {
            continue;
        }
        if (node.mesh >= static_cast<int>(model.meshes.size()))
        {
            throw std::runtime_error("GltfLoader: node references an invalid mesh");
        }
        if (node.skin >= static_cast<int>(model.skins.size()))
        {
            throw std::runtime_error("GltfLoader: node references an invalid skin");
        }

        instances.push_back({.sourceNodeIndex = nodeIndex,
                             .meshIndex = static_cast<uint32_t>(node.mesh),
                             .skinIndex = node.skin >= 0 ? std::optional<uint32_t>(node.skin) : std::nullopt,
                             .localTransform = extractNodeTransform(node)});
    }
    return instances;
}

glm::uvec4 readJointIndices(const AccessorView &view, size_t index)
{
    const unsigned char *data          = view.data + index * view.stride;
    const size_t         componentSize = tinygltf::GetComponentSizeInBytes(view.componentType);
    glm::uvec4           result{};
    for (size_t component = 0; component < 4; ++component)
    {
        const unsigned char *value = data + component * componentSize;
        result[component]          = view.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE
                                         ? *reinterpret_cast<const uint8_t *>(value)
                                         : *reinterpret_cast<const uint16_t *>(value);
    }
    return result;
}

void validateWeightView(const AccessorView &view)
{
    const bool validFloat = view.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && !view.normalized;
    const bool validNormalizedInteger =
        (view.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE ||
         view.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) &&
        view.normalized;
    if (view.type != TINYGLTF_TYPE_VEC4 || (!validFloat && !validNormalizedInteger))
    {
        throw std::runtime_error(
            "GltfLoader: WEIGHTS_n must be a VEC4 of floats or normalized unsigned bytes/shorts");
    }
}

struct InfluenceSetViews
{
    AccessorView joints;
    AccessorView weights;
};

std::vector<InfluenceSetViews> getInfluenceSetViews(const tinygltf::Primitive &primitive,
                                                    const tinygltf::Model &model, size_t vertexCount)
{
    std::vector<InfluenceSetViews> sets;
    for (size_t setIndex = 0;; ++setIndex)
    {
        const std::string jointsName  = "JOINTS_" + std::to_string(setIndex);
        const std::string weightsName = "WEIGHTS_" + std::to_string(setIndex);
        const auto        jointsIt    = primitive.attributes.find(jointsName);
        const auto        weightsIt   = primitive.attributes.find(weightsName);

        if (jointsIt == primitive.attributes.end() && weightsIt == primitive.attributes.end())
        {
            break;
        }
        if (jointsIt == primitive.attributes.end() || weightsIt == primitive.attributes.end())
        {
            throw std::runtime_error("GltfLoader: every JOINTS_n attribute must have a matching WEIGHTS_n");
        }

        const AccessorView jointsView = getAccessorView(model, jointsIt->second);
        const AccessorView weightsView = getAccessorView(model, weightsIt->second);
        validateJointView(jointsView);
        validateWeightView(weightsView);
        if (jointsView.count != vertexCount || weightsView.count != vertexCount)
        {
            throw std::runtime_error("GltfLoader: joint and weight accessor counts must match POSITION");
        }

        sets.push_back({jointsView, weightsView});
    }
    return sets;
}

std::vector<VertexGroupEntry> readVertexGroups(std::span<const InfluenceSetViews> sets, size_t vertexIndex)
{
    std::vector<VertexGroupEntry> groups;
    groups.reserve(sets.size() * 4);
    for (const InfluenceSetViews &set : sets)
    {
        const glm::uvec4 joints  = readJointIndices(set.joints, vertexIndex);
        const glm::vec4  weights = readVec<glm::vec4>(set.weights, vertexIndex);
        for (size_t component = 0; component < 4; ++component)
        {
            const float weight = weights[component];
            if (!std::isfinite(weight) || weight < 0.0f)
            {
                throw std::runtime_error("GltfLoader: joint weights must be finite and non-negative");
            }
            if (weight > 0.0f)
            {
                groups.push_back({joints[component], weight});
            }
        }
    }

    std::ranges::sort(groups, {}, &VertexGroupEntry::groupIndex);
    for (size_t i = 1; i < groups.size(); ++i)
    {
        if (groups[i - 1].groupIndex == groups[i].groupIndex)
        {
            throw std::runtime_error("GltfLoader: a joint may influence a vertex only once");
        }
    }

    float weightSum = 0.0f;
    for (const VertexGroupEntry &group : groups)
    {
        weightSum += group.weight;
    }
    if (weightSum > 0.0f)
    {
        for (VertexGroupEntry &group : groups)
        {
            group.weight /= weightSum;
        }
    }
    return groups;
}

static const std::string POSITION_ATTR_NAME = "POSITION";
static const std::string NORMAL_ATTR_NAME   = "NORMAL";
static const std::string UV_ATTR_NAME       = "TEXCOORD_0";
static const std::string TANGENT_ATTR_NAME  = "TANGENT";

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

struct PositionAndGroups
{
    glm::vec3                    position;
    std::vector<VertexGroupEntry> groups;

    bool operator==(const PositionAndGroups &) const = default;
};

struct PositionAndGroupsHash
{
    std::size_t operator()(const PositionAndGroups &key) const
    {
        std::size_t seed = Vector3Hash{}(key.position);
        auto combine = [&](std::size_t value) { seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2); };
        for (const VertexGroupEntry &group : key.groups)
        {
            combine(std::hash<uint32_t>{}(group.groupIndex));
            combine(std::hash<float>{}(group.weight));
        }
        return seed;
    }
};

struct ExtractedMeshData
{
    MeshData                                  geometry;
    std::vector<std::vector<VertexGroupEntry>> vertexGroups;
    bool                                      hasVertexGroups = false;
};

ExtractedMeshData extractMeshData(const tinygltf::Mesh &mesh, const tinygltf::Model &model,
                                  const std::vector<MaterialHandle> &materialHandles)
{
    ExtractedMeshData extracted;
    MeshData         &meshData = extracted.geometry;
    auto    &positions       = meshData.positions;
    auto    &positionIndices = meshData.positionIndices;
    auto    &normals         = meshData.normals;
    auto    &tangents        = meshData.tangents;
    auto    &uvs             = meshData.uvs;
    auto    &faces           = meshData.faces;
    auto    &faceGroups      = meshData.faceGroups;

    size_t totalVertexCount = 0;
    size_t totalFaceCount   = 0;

    bool hasTangents = true;

    for (const auto &primitive : mesh.primitives)
    {
        AccessorView posView = getAccessorView(model, primitive.attributes.at(POSITION_ATTR_NAME));
        totalVertexCount += posView.count;
        totalFaceCount +=
            primitive.indices >= 0 ? getAccessorView(model, primitive.indices).count / 3 : posView.count / 3;

        // We only use the provided tangents if they are present for every vertex. Otherwise, we will generate them
        // ourselves
        if (!primitive.attributes.contains(TANGENT_ATTR_NAME))
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

    positionIndices.reserve(totalVertexCount);
    normals.reserve(totalVertexCount);
    tangents.reserve(totalVertexCount);
    uvs.reserve(totalVertexCount);
    faces.reserve(totalFaceCount);
    faceGroups.reserve(totalFaceCount);

    // Generate the position indices
    std::unordered_map<PositionAndGroups, uint32_t, PositionAndGroupsHash> positionToIndex;

    // For each primitive, which is a submesh, not a triangle
    for (const tinygltf::Primitive &primitive : mesh.primitives)
    {
        size_t baseVertex = positionIndices.size();
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
        AccessorView posView     = getAccessorView(model, primitive.attributes.at(POSITION_ATTR_NAME));
        size_t       vertexCount = posView.count;
        const std::vector<InfluenceSetViews> influenceSets = getInfluenceSetViews(primitive, model, vertexCount);
        extracted.hasVertexGroups |= !influenceSets.empty();

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
            PositionAndGroups key{readVec<glm::vec3>(posView, j), readVertexGroups(influenceSets, j)};
            auto [it, inserted] = positionToIndex.try_emplace(key, static_cast<uint32_t>(positions.size()));
            if (inserted)
            {
                positions.push_back(key.position);
                extracted.vertexGroups.push_back(std::move(key.groups));
            }
            positionIndices.push_back(it->second);
        }

        // ATTRIBUTE 2 - Normal

        if (!primitive.attributes.contains(NORMAL_ATTR_NAME))
        {
            // If the normal accessor has fewer elements than the position accessor, we consider it as missing and fill
            // with default value.
            spdlog::warn(
                "GltfLoader: NORMAL attribute has fewer elements than POSITION attribute, filling with default value");
            for (size_t j = 0; j < vertexCount; ++j)
            {
                normals.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
            }
        } else
        {
            AccessorView normView = getAccessorView(model, primitive.attributes.at(NORMAL_ATTR_NAME));
            for (size_t j = 0; j < vertexCount; ++j)
            {
                normals.push_back(readVec<glm::vec3>(normView, j));
            }
        }

        // ATTRIBUTE 3 - UVs
        if (!primitive.attributes.contains(UV_ATTR_NAME))
        {
            // If the UV accessor has fewer elements than the position accessor, we consider it as missing and fill with
            // default value.
            spdlog::warn(
                "GltfLoader: {} attribute has fewer elements than POSITION attribute, filling with default value",
                UV_ATTR_NAME);
            for (size_t j = 0; j < vertexCount; ++j)
            {
                uvs.push_back(glm::vec2(0.0f, 0.0f));
            }
        } else
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
        } else
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
                faceGroups.push_back(primitive.material >= 0
                                         ? materialHandles.at(static_cast<size_t>(primitive.material) + 1)
                                         : materialHandles.at(0));
            }
        } else
        {
            AccessorView indexView = getAccessorView(model, primitive.indices);
            size_t       faceCount = static_cast<size_t>(indexView.count) / 3;
            for (size_t j = 0; j < faceCount; ++j)
            {
                const uint32_t i0 = readIndex(indexView, j * 3 + 0);
                const uint32_t i1 = readIndex(indexView, j * 3 + 1);
                const uint32_t i2 = readIndex(indexView, j * 3 + 2);
                faces.push_back(glm::uvec3(baseVertex + i0, baseVertex + i1, baseVertex + i2));

                faceGroups.push_back(primitive.material >= 0
                                         ? materialHandles.at(static_cast<size_t>(primitive.material) + 1)
                                         : materialHandles.at(0));
            }
        }
    }

    // Generate tangents using mikktspace.

    if (!hasTangents)
    {
        tangents.resize(positionIndices.size(), glm::vec4(0.0f));
        generateTangents(meshData);
    }

    return extracted;
}

} // namespace

GltfMeshLoadResult GltfLoader::load(const std::filesystem::path &path, MaterialStore &materialStore,
                                    const GltfLoaderConfig &config) const
{
    // SECTION 1 - Load the glTF file using tinygltf to obtain a tinygltf::Model instance.

    if (path.empty())
    {
        throw std::invalid_argument("GltfLoader: empty path");
    }

    tinygltf::Model model = loadGltfFile(path);

    // SECTION 2 - Extract material data and register it into the store first — mesh extraction
    // below needs the resulting handles to bake directly into each face's faceGroups entry.
    std::vector<MaterialHandle> materialHandles = extractMaterials(model, config, materialStore);

    // SECTION 3 - Extract vertex / face data from the tinygltf::Model

    MeshSequence seq;
    seq.frames.reserve(model.meshes.size());
    for (size_t i = 0; i < model.meshes.size(); ++i)
    {
        Mesh                 &outMesh = seq.frames.emplace_back();
        const tinygltf::Mesh &mesh    = model.meshes[i];

        if (mesh.primitives.empty())
        {
            throw std::runtime_error("GltfLoader: mesh has no primitives");
        }

        ExtractedMeshData extracted = extractMeshData(mesh, model, materialHandles);
        MeshData         &geometry  = extracted.geometry;

        if (extracted.hasVertexGroups)
        {
            outMesh.enableVertexGroups();
        }
        outMesh.setTopology(std::move(geometry.positions), std::move(geometry.positionIndices),
                            std::move(geometry.faces));
        outMesh.setFaceGroups(std::move(geometry.faceGroups));

        if (extracted.hasVertexGroups)
        {
            for (uint32_t vertex = 0; vertex < extracted.vertexGroups.size(); ++vertex)
            {
                outMesh.setVertexGroups(vertex, extracted.vertexGroups[vertex]);
            }
        }

        outMesh.setPerVertexArray<glm::vec3>(config.normalAttributeName, geometry.normals);
        outMesh.setPerVertexArray<glm::vec4>(config.tangentAttributeName, geometry.tangents);
        outMesh.setPerVertexArray<glm::vec2>(config.uvAttributeName, geometry.uvs);
    }

    const std::vector<std::optional<uint32_t>> sourceParents = buildNodeParents(model);
    std::vector<Skin>                          skins         = extractSkins(model, sourceParents);
    std::vector<GltfMeshInstance>              meshInstances = extractMeshInstances(model);

    for (const GltfMeshInstance &instance : meshInstances)
    {
        if (!instance.skinIndex)
        {
            continue;
        }

        const Mesh &mesh = seq.frames[instance.meshIndex];
        const Skin &skin = skins[instance.skinIndex.value()];
        if (!mesh.layout().vertexGroupsEnabled())
        {
            continue;
        }
        for (uint32_t vertexIndex = 0; vertexIndex < mesh.uniquePositionCount(); ++vertexIndex)
        {
            for (const VertexGroupEntry &influence : mesh.getVertexGroups(vertexIndex))
            {
                if (influence.groupIndex >= skin.joints().size())
                {
                    throw std::runtime_error("GltfLoader: mesh influence references a joint outside its skin palette");
                }
            }
        }
    }

    return {std::move(seq), std::move(materialHandles), std::move(skins), std::move(meshInstances)};
}

} // namespace lr
