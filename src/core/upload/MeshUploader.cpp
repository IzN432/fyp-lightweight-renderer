#include "MeshUploader.hpp"

#include "core/vulkan/VkFormatUtils.hpp"

namespace lr
{

namespace
{


struct PackedVertexAttributes
{
    // Buffers containing the packed vertex attributes
    std::unordered_map<uint32_t, std::vector<std::byte>> vertexAttributeBufferMap;
};

PackedVertexAttributes packVertexAttributes(const Mesh &mesh, const GpuMeshLayout &gpuLayout)
{
    PackedVertexAttributes out;
    const auto &mappings = gpuLayout.mappings();
    const uint32_t vertexCount = mesh.vertexCount();

    out.vertexAttributeBufferMap.reserve(gpuLayout.bindingDescriptions().size());
    const auto &bindingDescriptions = gpuLayout.bindingDescriptions();
    const auto &attributeDescriptions = gpuLayout.attributeDescriptions();

    std::unordered_map<uint32_t, uint32_t> bindingToStride;
    for (const auto &binding : bindingDescriptions)
    {
        bindingToStride[binding.binding] = binding.stride;
    }

    for (const auto &description : bindingDescriptions)
    {
        auto &buffer = out.vertexAttributeBufferMap[description.binding];
        buffer.resize(static_cast<size_t>(vertexCount) * description.stride);
    }

    for (size_t i = 0; i < mappings.size(); ++i)
    {
        const auto &mapping = mappings[i];
        const uint32_t binding = mapping.binding;
        const auto &attributeDescription = attributeDescriptions[i];

        std::span<const std::byte> data;
        if (mapping.isPosition)
        {
            // Positions are a special case since they are stored explicitly as glm::vec3.
            // So, we explicitly convert it into a span of bytes here.
            data = { reinterpret_cast<const std::byte *>(mesh.positions.data()),
                    mesh.positions.size() * sizeof(glm::vec3) };
        }
        else
        {
            data = mesh.rawPerVertexData(mapping.name);
        }

        const size_t currentStride = bindingToStride[binding];
        const size_t currentAttributeOffset = attributeDescription.offset;
        const size_t attributeSize = vkFormatByteSize(mapping.format);
        if (data.size() < static_cast<size_t>(vertexCount) * attributeSize)
        {
            throw std::runtime_error(
                "MeshUploader: attribute '" + mapping.name + "' has fewer elements than vertexCount");
        }

        auto &buffer = out.vertexAttributeBufferMap[binding];

        for (uint32_t v = 0; v < vertexCount; ++v)
        {
            const std::byte *src = data.data() + (v * attributeSize);
            std::byte *dst = buffer.data() + (v * currentStride) + currentAttributeOffset;
            std::memcpy(dst, src, attributeSize);
        }
    }

    return out;
}

} // namespace

MeshUploader::MeshUploader(ResourceRegistry &registry)
    : m_registry(registry)
{
}

MeshUploadResult MeshUploader::upload(const Mesh &mesh,
                        const GpuMeshLayout &gpuLayout,
                        const std::string &namePrefix)
{
    MeshUploadResult result;
    
    // SECTION 1 - Add per-vertex attributes to GPU buffers

    const auto packed = packVertexAttributes(mesh, gpuLayout);

    for (const auto &[binding, bytes] : packed.vertexAttributeBufferMap)
    {
        const std::string name = namePrefix + "_vb_" + std::to_string(binding);
        m_registry.uploadBuffer(name,
                                bytes.data(),
                                static_cast<VkDeviceSize>(bytes.size()),
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        result.vertexBufferNames.emplace(binding, name);
    }

    // SECTION 2 - Add indices to GPU buffer

    std::vector<std::byte> indexBuffer;

    indexBuffer.resize(static_cast<size_t>(mesh.faces.size()) * sizeof(uint32_t) * 3);
    std::memcpy(indexBuffer.data(), mesh.faces.data(), indexBuffer.size());

    const std::string indexName = namePrefix + "_ib";
    m_registry.uploadBuffer(indexName,
                            indexBuffer.data(),
                            static_cast<VkDeviceSize>(indexBuffer.size()),
                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    result.indexBufferName = indexName;
    result.indexCount = mesh.faceCount() * 3;

    // SECTION 3 - Add vertex groups to GPU buffer
    // TODO: Add vertex groups including weights

    // SECTION 4 - Add face group indices to GPU buffer (optional)
    if (!mesh.faceGroups.empty())
    {
        std::vector<std::byte> faceGroupBuffer;
    
        faceGroupBuffer.resize(static_cast<size_t>(mesh.faceGroups.size()) * sizeof(uint32_t));
        std::memcpy(faceGroupBuffer.data(), mesh.faceGroups.data(), faceGroupBuffer.size());
    
        const std::string faceGroupName = namePrefix + "_fg";
        m_registry.uploadBuffer(faceGroupName,
                                faceGroupBuffer.data(),
                                static_cast<VkDeviceSize>(faceGroupBuffer.size()),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        result.faceGroupBufferName = faceGroupName;
    }

    return result;
} 

} // namespace lr