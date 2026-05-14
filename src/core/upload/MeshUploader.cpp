#include "MeshUploader.hpp"

#include "core/vulkan/VkFormatUtils.hpp"

#include <stdexcept>

namespace lr
{

namespace
{


struct VertexBuffer
{
    std::vector<std::byte> vertexAttributeBuffer;
};

// Packs the vertex attributes of the provided meshes into a contiguous buffer
VertexBuffer packVertexAttributes(const std::vector<const Mesh*> &meshes, const VertexBufferUploadConfig &config)
{
    if (meshes.empty())
        throw std::invalid_argument("packVertexAttributes: meshes must not be empty");

    VertexBuffer out;

    const MeshLayout &layout = meshes[0]->layout();

    uint32_t stride = 0;
    for (const auto &attribute : config.vertexAttributeNames)
    {
        const auto *attr = layout.findPerVertexAttr(attribute);
        if (!attr)
            throw std::invalid_argument("packVertexAttributes: attribute '" + attribute + "' not found in mesh layout");
        stride += attr->stride;
    }
    if (config.includePosition)
    {
        stride += sizeof(glm::vec3);
    }

    uint32_t vertexCount = 0;
    for (const auto &mesh : meshes)
    {
        vertexCount += mesh->vertexCount();
    }

    out.vertexAttributeBuffer.resize(static_cast<size_t>(vertexCount) * stride);

    // Assign each attribute to an offset, based on the ordering in the config
    std::unordered_map<std::string, uint32_t> attributeOffsets;
    std::unordered_map<std::string, uint32_t> attributeStrides;
    uint32_t currentOffset = 0;

    // If position is included, it goes to the start of the vertex data
    if (config.includePosition)
    {
        currentOffset += sizeof(glm::vec3);
    }
    for (const auto &attribute : config.vertexAttributeNames)
    {
        const auto *attr = layout.findPerVertexAttr(attribute);
        if (!attr)
            throw std::invalid_argument("packVertexAttributes: attribute '" + attribute + "' not found in mesh layout");
        attributeOffsets[attribute] = currentOffset;
        attributeStrides[attribute] = attr->stride;
        currentOffset += attr->stride;
    }

    uint32_t vertexOffset = 0;
    for (size_t i = 0; i < meshes.size(); ++i)
    {
        const auto &mesh = *meshes[i];

        auto uploadData = [&](std::span<const std::byte> data, uint32_t attributeStride, uint32_t attributeOffset)
        {
            for (uint32_t v = 0; v < mesh.vertexCount(); ++v)
            {
                std::byte *vertexPtr = out.vertexAttributeBuffer.data() + ((vertexOffset + v) * stride);
                std::byte *dst = vertexPtr + attributeOffset;
                const std::byte *src = data.data() + (v * attributeStride);
                std::memcpy(dst, src, attributeStride);
            }
        };

        if (config.includePosition)
        {
            auto positionData = reinterpret_cast<const std::byte*>(mesh.positions.data());
            std::span<const std::byte> data(positionData, mesh.vertexCount() * sizeof(glm::vec3));
            uploadData(data, sizeof(glm::vec3), 0);
        }

        for (const auto &attribute : config.vertexAttributeNames)
        {
            // Get the attribute data
            std::span<const std::byte> attributeData = mesh.rawPerVertexData(attribute);
            uploadData(attributeData, attributeStrides[attribute], attributeOffsets[attribute]);
        }

        vertexOffset += mesh.vertexCount();
    }

    return out;
}

} // namespace

MeshUploader::MeshUploader(ResourceRegistry &registry)
    : m_registry(registry)
{
}

VertexBufferUploadResult MeshUploader::uploadVertexBuffer(const std::vector<const Mesh*> &meshes,
                                                          const VertexBufferUploadConfig &config)
{
    // We assume that all the meshes have the same layout, so we can use the first one to get the layout information.
    // In a more robust implementation, we would want to check that all meshes have the same layout

    VertexBufferUploadResult result;

    const auto packed = packVertexAttributes(meshes, config);

    const std::string name = config.vertexBufferName;
    m_registry.uploadBuffer(name,
                            packed.vertexAttributeBuffer.data(),
                            static_cast<VkDeviceSize>(packed.vertexAttributeBuffer.size()),
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    uint32_t vertexOffset = 0;
    for (const auto &mesh : meshes)
    {
        result.singleMeshResults.push_back({ .vertexOffset = vertexOffset });
        vertexOffset += mesh->vertexCount();
    }
    
    return result;
} 

IndexBufferUploadResult MeshUploader::uploadIndexBuffer(const std::vector<const Mesh*> &meshes, const IndexBufferUploadConfig &config)
{
    IndexBufferUploadResult result;

    uint32_t totalFaceCount = 0;
    for (const auto &mesh : meshes)
    {
        totalFaceCount += mesh->faceCount();
    }

    std::vector<std::byte> indexBuffer;
    indexBuffer.resize(static_cast<size_t>(totalFaceCount) * sizeof(glm::uvec3));

    uint32_t faceOffset = 0;
    for (const auto &mesh : meshes)
    {
        result.singleMeshResults.push_back({
            .firstIndex = faceOffset * 3, // 3 indices per face
            .indexCount = mesh->faceCount() * 3
        });
        std::memcpy(indexBuffer.data() + faceOffset * sizeof(glm::uvec3),
                    mesh->faces.data(),
                    mesh->faceCount() * sizeof(glm::uvec3));
        faceOffset += mesh->faceCount();
    }

    const std::string name = config.indexBufferName;
    m_registry.uploadBuffer(name,
                            indexBuffer.data(),
                            static_cast<VkDeviceSize>(indexBuffer.size()),
                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    return result;
}

void MeshUploader::uploadFaceGroupBuffer(const std::vector<const Mesh*> &meshes, const FaceGroupBufferUploadConfig &config)
{
    bool anyFaceGroups = false;
    uint32_t totalFaceCount = 0;
    for (const auto &mesh : meshes)
    {
        totalFaceCount += mesh->faceCount();
        anyFaceGroups |= !mesh->faceGroups.empty();
    }

    if (!anyFaceGroups)
    {
        throw std::invalid_argument("uploadFaceGroupBuffer: meshes must have face groups");
    }

    std::vector<std::byte> faceGroupBuffer;
    faceGroupBuffer.resize(static_cast<size_t>(totalFaceCount) * sizeof(uint32_t));

    uint32_t faceOffset = 0;
    for (const auto &mesh : meshes)
    {
        if (mesh->faceGroups.empty())
        {
            std::memset(faceGroupBuffer.data() + faceOffset * sizeof(uint32_t), 0, mesh->faceCount() * sizeof(uint32_t));
        }
        else
        {
            std::memcpy(faceGroupBuffer.data() + faceOffset * sizeof(uint32_t),
                        mesh->faceGroups.data(),
                        mesh->faceGroups.size() * sizeof(uint32_t));
        }
        faceOffset += mesh->faceCount();
    }

    const std::string name = config.faceGroupBufferName;
    m_registry.uploadBuffer(name,
                            faceGroupBuffer.data(),
                            static_cast<VkDeviceSize>(faceGroupBuffer.size()),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

void MeshUploader::uploadVertexGroupBuffers(const std::vector<const Mesh*> &meshes,
                                            const VertexGroupBufferUploadConfig &config)
{
    bool anyVertexGroups = false;
    uint32_t totalVertexCount = 0;
    uint32_t totalEntryCount  = 0;
    for (const auto &mesh : meshes)
    {
        totalVertexCount += mesh->vertexCount();
        if (mesh->layout().vertexGroupsEnabled())
        {
            anyVertexGroups = true;
            totalEntryCount += static_cast<uint32_t>(mesh->rawGroupEntries().size());
        }
    }

    if (!anyVertexGroups)
        throw std::invalid_argument("uploadVertexGroupBuffers: meshes must have vertex groups enabled");

    // Guarantee a non-zero entries buffer (Vulkan disallows zero-size storage buffers)
    std::vector<VertexGroupEntry> entries(std::max(totalEntryCount, 1u));
    std::vector<uint32_t>         offsets(totalVertexCount, 0u);
    std::vector<uint32_t>         counts(totalVertexCount,  0u);

    uint32_t vertexCursor = 0;
    uint32_t entryCursor  = 0;
    for (const auto &mesh : meshes)
    {
        const uint32_t vCount = mesh->vertexCount();
        if (mesh->layout().vertexGroupsEnabled())
        {
            auto meshEntries = mesh->rawGroupEntries();
            auto meshOffsets = mesh->rawGroupOffsets();
            auto meshCounts  = mesh->rawGroupCounts();

            std::memcpy(entries.data() + entryCursor,
                        meshEntries.data(),
                        meshEntries.size() * sizeof(VertexGroupEntry));

            for (uint32_t v = 0; v < vCount; ++v)
            {
                offsets[vertexCursor + v] = meshOffsets[v] + entryCursor;
                counts [vertexCursor + v] = meshCounts[v];
            }

            entryCursor += static_cast<uint32_t>(meshEntries.size());
        }
        // meshes without vertex groups: offsets/counts stay zero-initialised

        vertexCursor += vCount;
    }

    m_registry.uploadBuffer(config.entriesBufferName,
                            entries.data(),
                            static_cast<VkDeviceSize>(entries.size() * sizeof(VertexGroupEntry)),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    m_registry.uploadBuffer(config.offsetsBufferName,
                            offsets.data(),
                            static_cast<VkDeviceSize>(offsets.size() * sizeof(uint32_t)),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    m_registry.uploadBuffer(config.countsBufferName,
                            counts.data(),
                            static_cast<VkDeviceSize>(counts.size() * sizeof(uint32_t)),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

} // namespace lr