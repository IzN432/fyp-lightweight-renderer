#pragma once

#include "core/upload/MeshBufferCache.hpp"

#include <cstring>

namespace lr
{

// Pure CPU packing shared by initial uploads, synchronization, and replacements.
// Domain expansion here keeps every GPU view a projection of canonical mesh data.
inline std::vector<std::byte> packMeshVertexData(const std::vector<const Mesh *> &meshes,
                                               const VertexBufferUploadConfig &config,
                                               bool uniqueVertices = false)
{
    // An empty mesh list packs to an empty payload: the stride is unknowable without a mesh to read
    // it from, but with no vertices to write there is nothing for it to describe. This is a normal
    // state (a scene with no geometry yet), and ResourceRegistry/Allocator already accept a
    // zero-sized payload, so the GPU views still exist for passes to bind.
    if (meshes.empty()) return {};
    const bool uniqueAttributes = uniqueVertices || config.expandUniqueVertexAttributes;
    const auto findAttribute = [&](const Mesh &mesh, const std::string &name) {
        const auto *desc = uniqueAttributes ? mesh.layout().findPerUniqueVertexAttr(name)
                                           : mesh.layout().findPerVertexAttr(name);
        if (!desc) throw std::invalid_argument("Mesh packing: missing attribute '" + name + "'");
        return desc;
    };

    size_t stride = config.includePosition ? sizeof(glm::vec3) : 0;
    for (const auto &name : config.vertexAttributeNames) stride += findAttribute(*meshes.front(), name)->stride;
    size_t totalCount = 0;
    for (const Mesh *mesh : meshes)
    {
        totalCount += uniqueVertices ? mesh->uniquePositionCount() : mesh->vertexCount();
        for (const auto &name : config.vertexAttributeNames)
        {
            const auto *expected = findAttribute(*meshes.front(), name);
            const auto *actual = findAttribute(*mesh, name);
            if (actual->stride != expected->stride || actual->type != expected->type)
                throw std::invalid_argument("Mesh packing: incompatible attribute '" + name + "'");
        }
    }

    std::vector<std::byte> packed(totalCount * stride);
    size_t vertexOffset = 0;
    for (const Mesh *mesh : meshes)
    {
        const uint32_t count = uniqueVertices ? mesh->uniquePositionCount() : mesh->vertexCount();
        if (config.includePosition)
        {
            for (uint32_t vertex = 0; vertex < count; ++vertex)
            {
                const uint32_t position = uniqueVertices ? vertex : mesh->positionIndices()[vertex];
                std::memcpy(packed.data() + (vertexOffset + vertex) * stride,
                            &mesh->positions()[position], sizeof(glm::vec3));
            }
        }
        size_t attributeOffset = config.includePosition ? sizeof(glm::vec3) : 0;
        for (const auto &name : config.vertexAttributeNames)
        {
            const size_t attributeStride = findAttribute(*mesh, name)->stride;
            const auto data = uniqueAttributes ? mesh->rawPerUniqueVertexData(name) : mesh->rawPerVertexData(name);
            const size_t sourceCount = uniqueAttributes ? mesh->uniquePositionCount() : mesh->vertexCount();
            if (data.size() != sourceCount * attributeStride)
                throw std::invalid_argument("Mesh packing: incomplete attribute '" + name + "'");
            for (uint32_t vertex = 0; vertex < count; ++vertex)
            {
                const uint32_t source = !uniqueVertices && uniqueAttributes ? mesh->positionIndices()[vertex] : vertex;
                std::memcpy(packed.data() + (vertexOffset + vertex) * stride + attributeOffset,
                            data.data() + source * attributeStride, attributeStride);
            }
            attributeOffset += attributeStride;
        }
        vertexOffset += count;
    }
    return packed;
}

} // namespace lr
