#include "core/upload/MeshUploader.hpp"

#include "core/utility/ImageLoader.hpp"
#include "core/vulkan/VkFormatUtils.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <unordered_map>
#include <vector>

namespace lr
{

namespace
{

// Interleaved vertex + index packing — formerly MeshPacker.
struct PackedMesh
{
    std::unordered_map<uint32_t, std::vector<std::byte>> vertexData;
    std::vector<std::byte> indexData;
    uint32_t indexCount = 0;
};

PackedMesh packMesh(const Mesh &mesh, const GpuMeshLayout &layout)
{
    const auto    &mappings    = layout.mappings();
    const uint32_t vertexCount = mesh.vertexCount();

    struct MappingInfo { size_t mappingIdx; uint32_t offsetInStride; };
    std::unordered_map<uint32_t, std::vector<MappingInfo>> byBinding;
    std::vector<uint32_t> bindingOrder;

    for (size_t i = 0; i < mappings.size(); ++i)
    {
        const uint32_t b = mappings[i].binding;
        if (byBinding.find(b) == byBinding.end())
            bindingOrder.push_back(b);

        uint32_t offset = 0;
        for (const auto &mi : byBinding[b])
            offset += vkFormatByteSize(mappings[mi.mappingIdx].format);

        byBinding[b].push_back({i, offset});
    }

    PackedMesh result;

    for (uint32_t binding : bindingOrder)
    {
        const auto &infos = byBinding.at(binding);

        uint32_t stride = 0;
        for (const auto &mi : infos)
            stride += vkFormatByteSize(mappings[mi.mappingIdx].format);

        std::vector<std::byte> cpuBuf(static_cast<size_t>(vertexCount) * stride);

        for (const auto &mi : infos)
        {
            const auto    &m        = mappings[mi.mappingIdx];
            const uint32_t attrSize = vkFormatByteSize(m.format);

            std::span<const std::byte> src;
            if (m.isPosition)
                src = { reinterpret_cast<const std::byte *>(mesh.positions.data()),
                        mesh.positions.size() * sizeof(glm::vec3) };
            else
                src = mesh.rawPerVertexData(m.name);

            if (src.size() < static_cast<size_t>(vertexCount) * attrSize)
                throw std::runtime_error(
                    "MeshUploader: attribute '" + m.name + "' has fewer elements than vertexCount");

            for (uint32_t v = 0; v < vertexCount; ++v)
            {
                std::byte       *dst = cpuBuf.data() + v * stride + mi.offsetInStride;
                const std::byte *s   = src.data()    + v * attrSize;
                std::memcpy(dst, s, attrSize);
            }
        }

        result.vertexData[binding] = std::move(cpuBuf);
    }

    const uint32_t indexCount = mesh.faceCount() * 3;
    result.indexData.resize(static_cast<size_t>(indexCount) * sizeof(uint32_t));
    std::memcpy(result.indexData.data(), mesh.faces.data(), result.indexData.size());
    result.indexCount = indexCount;

    return result;
}

} // namespace

MeshUploader::MeshUploader(ResourceRegistry &registry)
    : m_registry(registry)
{
}

MeshUploadResult MeshUploader::upload(const Mesh &mesh,
                                      const GpuMeshLayout &gpuLayout,
                                      const std::vector<MaterialInfo> &materials,
                                      const std::string &namePrefix)
{
    MeshUploadResult out;

    // -----------------------------------
    // Vertex and index buffers
    // -----------------------------------
    const PackedMesh packed = packMesh(mesh, gpuLayout);

    for (const auto &[binding, bytes] : packed.vertexData)
    {
        const std::string name = namePrefix + "_vb_" + std::to_string(binding);
        m_registry.uploadBuffer(name,
                                bytes.data(),
                                static_cast<VkDeviceSize>(bytes.size()),
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        out.vertexBufferNames.emplace(binding, name);
    }

    out.indexBufferName = namePrefix + "_ib";
    m_registry.uploadBuffer(out.indexBufferName,
                            packed.indexData.data(),
                            static_cast<VkDeviceSize>(packed.indexData.size()),
                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    out.indexCount = packed.indexCount;

    // -----------------------------------
    // Per-face group index SSBO
    // -----------------------------------
    std::vector<uint32_t> faceGroupIndices(mesh.faceCount(), 0u);
    if (mesh.faceGroups.size() == mesh.faceCount())
        faceGroupIndices = mesh.faceGroups;

    out.faceGroupIndexBufferName = namePrefix + "_face_group_indices";
    m_registry.uploadBuffer(out.faceGroupIndexBufferName,
                            faceGroupIndices.data(),
                            static_cast<VkDeviceSize>(faceGroupIndices.size() * sizeof(uint32_t)),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // -----------------------------------
    // Texture arrays (one slot per material)
    // -----------------------------------
    out.diffuseTextureArrayName   = namePrefix + "_diffuse_array";
    out.specularTextureArrayName  = namePrefix + "_specular_array";
    out.normalTextureArrayName    = namePrefix + "_normal_array";
    out.roughnessTextureArrayName = namePrefix + "_roughness_array";
    out.metallicTextureArrayName  = namePrefix + "_metallic_array";
    out.emissiveTextureArrayName  = namePrefix + "_emissive_array";

    const std::array<uint8_t, 4> white           = {255, 255, 255, 255};  // diffuse/basecolor
    const std::array<uint8_t, 4> black           = {0,   0,   0,   255};  // metallic, emissive
    const std::array<uint8_t, 4> neutralNormal   = {128, 128, 255, 255};  // flat normal (0.5, 0.5, 1.0)
    const std::array<uint8_t, 4> mediumRoughness = {128, 128, 128, 255};  // 0.5 roughness

    uint32_t maxGroup = 0u;
    for (uint32_t g : faceGroupIndices)
        maxGroup = std::max(maxGroup, g);

    const uint32_t materialCount = static_cast<uint32_t>(materials.size());
    out.materialTextureCount = std::max(1u, std::max(materialCount, maxGroup + 1u));

    auto uploadTextureArray = [&](const std::string &arrayName,
                                  auto getPath,
                                  const std::array<uint8_t, 4> &fallback)
    {
        if (materials.empty())
        {
            m_registry.uploadArrayImage(arrayName, 0,
                                        fallback.data(), 1, 1,
                                        VK_FORMAT_R8G8B8A8_SRGB);
            return;
        }

        for (uint32_t i = 0; i < out.materialTextureCount; ++i)
        {
            const std::filesystem::path tex = (i < materialCount)
                ? getPath(materials[i].textures)
                : std::filesystem::path{};

            if (!tex.empty() && std::filesystem::exists(tex))
            {
                LoadedImage img = loadImageFromFile(tex);
                m_registry.uploadArrayImage(arrayName, i,
                                            img.pixels, img.width, img.height,
                                            LoadedImage::format, true);
            }
            else
            {
                m_registry.uploadArrayImage(arrayName, i,
                                            fallback.data(), 1, 1,
                                            VK_FORMAT_R8G8B8A8_SRGB);
            }
        }
    };

    uploadTextureArray(out.diffuseTextureArrayName,
        [](const MaterialTextures &t) -> const std::filesystem::path & { return t.baseColor; },
        white);
    uploadTextureArray(out.specularTextureArrayName,
        [](const MaterialTextures &t) -> const std::filesystem::path & { return t.specular; },
        white);
    uploadTextureArray(out.normalTextureArrayName,
        [](const MaterialTextures &t) -> const std::filesystem::path & { return t.normal; },
        neutralNormal);
    uploadTextureArray(out.roughnessTextureArrayName,
        [](const MaterialTextures &t) -> const std::filesystem::path & { return t.roughness; },
        mediumRoughness);
    uploadTextureArray(out.metallicTextureArrayName,
        [](const MaterialTextures &t) -> const std::filesystem::path & { return t.metallic; },
        black);
    uploadTextureArray(out.emissiveTextureArrayName,
        [](const MaterialTextures &t) -> const std::filesystem::path & { return t.emissive; },
        black);

    return out;
}

} // namespace lr
