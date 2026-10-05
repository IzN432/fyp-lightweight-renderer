#include "MeshUploader.hpp"

#include "core/upload/MeshBufferPacking.hpp"

namespace lr
{
namespace
{
VertexBufferUploadResult vertexRanges(const std::vector<const Mesh *> &meshes, bool uniqueVertices = false)
{
    VertexBufferUploadResult result;
    uint32_t offset = 0;
    for (const Mesh *mesh : meshes)
    {
        result.singleMeshResults.push_back({.vertexOffset = offset});
        offset += uniqueVertices ? mesh->uniquePositionCount() : mesh->vertexCount();
    }
    return result;
}

struct PackedIndices
{
    std::vector<glm::uvec3> data;
    IndexBufferUploadResult ranges;
};

PackedIndices packIndices(const std::vector<const Mesh *> &meshes)
{
    PackedIndices packed;
    for (const Mesh *mesh : meshes)
    {
        packed.ranges.singleMeshResults.push_back({.firstIndex = static_cast<uint32_t>(packed.data.size()) * 3,
                                                  .indexCount = mesh->faceCount() * 3});
        packed.data.insert(packed.data.end(), mesh->faces().begin(), mesh->faces().end());
    }
    return packed;
}

std::vector<uint32_t> packFaceGroups(const std::vector<const Mesh *> &meshes)
{
    std::vector<uint32_t> packed;
    for (const Mesh *mesh : meshes)
    {
        if (mesh->faceGroups().empty()) packed.insert(packed.end(), mesh->faceCount(), 0);
        else packed.insert(packed.end(), mesh->faceGroups().begin(), mesh->faceGroups().end());
    }
    return packed;
}
} // namespace

MeshUploader::MeshUploader(ResourceRegistry &registry) : m_registry(registry) {}

VertexBufferUploadResult MeshUploader::uploadVertexBuffer(const std::vector<const Mesh *> &meshes,
                                                          const VertexBufferUploadConfig &config)
{
    auto stamp = MeshBufferCache::vertices(meshes, config);
    const auto packed = packMeshVertexData(meshes, config);
    m_registry.uploadBuffer(config.vertexBufferName, packed.data(), packed.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m_bufferCache.remember(config.vertexBufferName, std::move(stamp));
    return vertexRanges(meshes);
}

bool MeshUploader::synchronizeVertexBuffer(const std::vector<const Mesh *> &meshes,
                                          const VertexBufferUploadConfig &config)
{
    return m_bufferCache.synchronize(config.vertexBufferName, MeshBufferCache::vertices(meshes, config), [&]() {
        const auto packed = packMeshVertexData(meshes, config);
        m_registry.reuploadBuffer(config.vertexBufferName, packed.data(), packed.size());
    });
}

VertexBufferUploadResult MeshUploader::replaceVertexBuffer(const std::vector<const Mesh *> &meshes,
                                                           const VertexBufferUploadConfig &config)
{
    auto stamp = MeshBufferCache::vertices(meshes, config);
    const auto packed = packMeshVertexData(meshes, config);
    m_registry.replaceUploadedBuffer(config.vertexBufferName, packed.data(), packed.size(),
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m_bufferCache.remember(config.vertexBufferName, std::move(stamp));
    return vertexRanges(meshes);
}

VertexBufferUploadResult MeshUploader::uploadUniqueVertexBuffer(const std::vector<const Mesh *> &meshes,
                                                                const VertexBufferUploadConfig &config)
{
    auto stamp = MeshBufferCache::vertices(meshes, config, true);
    const auto packed = packMeshVertexData(meshes, config, true);
    m_registry.uploadBuffer(config.vertexBufferName, packed.data(), packed.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m_bufferCache.remember(config.vertexBufferName, std::move(stamp));
    return vertexRanges(meshes, true);
}

bool MeshUploader::synchronizeUniqueVertexBuffer(const std::vector<const Mesh *> &meshes,
                                                const VertexBufferUploadConfig &config)
{
    return m_bufferCache.synchronize(config.vertexBufferName, MeshBufferCache::vertices(meshes, config, true), [&]() {
        const auto packed = packMeshVertexData(meshes, config, true);
        m_registry.reuploadBuffer(config.vertexBufferName, packed.data(), packed.size());
    });
}

VertexBufferUploadResult MeshUploader::replaceUniqueVertexBuffer(const std::vector<const Mesh *> &meshes,
                                                                 const VertexBufferUploadConfig &config)
{
    auto stamp = MeshBufferCache::vertices(meshes, config, true);
    const auto packed = packMeshVertexData(meshes, config, true);
    m_registry.replaceUploadedBuffer(config.vertexBufferName, packed.data(), packed.size(),
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m_bufferCache.remember(config.vertexBufferName, std::move(stamp));
    return vertexRanges(meshes, true);
}

IndexBufferUploadResult MeshUploader::uploadIndexBuffer(const std::vector<const Mesh *> &meshes,
                                                       const IndexBufferUploadConfig &config)
{
    auto stamp = MeshBufferCache::indices(meshes);
    auto packed = packIndices(meshes);
    m_registry.uploadBuffer(config.indexBufferName, packed.data.data(), packed.data.size() * sizeof(glm::uvec3),
                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    m_bufferCache.remember(config.indexBufferName, std::move(stamp));
    return std::move(packed.ranges);
}

IndexBufferUploadResult MeshUploader::replaceIndexBuffer(const std::vector<const Mesh *> &meshes,
                                                        const IndexBufferUploadConfig &config)
{
    auto stamp = MeshBufferCache::indices(meshes);
    auto packed = packIndices(meshes);
    m_registry.replaceUploadedBuffer(config.indexBufferName, packed.data.data(), packed.data.size() * sizeof(glm::uvec3),
                                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    m_bufferCache.remember(config.indexBufferName, std::move(stamp));
    return std::move(packed.ranges);
}

bool MeshUploader::synchronizeIndexBuffer(const std::vector<const Mesh *> &meshes, const IndexBufferUploadConfig &config)
{
    return m_bufferCache.synchronize(config.indexBufferName, MeshBufferCache::indices(meshes), [&]() {
        const auto packed = packIndices(meshes);
        m_registry.reuploadBuffer(config.indexBufferName, packed.data.data(), packed.data.size() * sizeof(glm::uvec3));
    });
}

void MeshUploader::uploadFaceGroupBuffer(const std::vector<const Mesh *> &meshes, const FaceGroupBufferUploadConfig &config)
{
    auto stamp = MeshBufferCache::faceGroups(meshes);
    const auto packed = packFaceGroups(meshes);
    m_registry.uploadBuffer(config.faceGroupBufferName, packed.data(), packed.size() * sizeof(uint32_t),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_bufferCache.remember(config.faceGroupBufferName, std::move(stamp));
}

void MeshUploader::replaceFaceGroupBuffer(const std::vector<const Mesh *> &meshes,
                                         const FaceGroupBufferUploadConfig &config)
{
    auto stamp = MeshBufferCache::faceGroups(meshes);
    const auto packed = packFaceGroups(meshes);
    m_registry.replaceUploadedBuffer(config.faceGroupBufferName, packed.data(), packed.size() * sizeof(uint32_t),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_bufferCache.remember(config.faceGroupBufferName, std::move(stamp));
}

bool MeshUploader::synchronizeFaceGroupBuffer(const std::vector<const Mesh *> &meshes,
                                             const FaceGroupBufferUploadConfig &config)
{
    return m_bufferCache.synchronize(config.faceGroupBufferName, MeshBufferCache::faceGroups(meshes), [&]() {
        const auto packed = packFaceGroups(meshes);
        m_registry.reuploadBuffer(config.faceGroupBufferName, packed.data(), packed.size() * sizeof(uint32_t));
    });
}

} // namespace lr
