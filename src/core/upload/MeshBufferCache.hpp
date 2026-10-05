#pragma once

#include "core/scene/Mesh.hpp"

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lr
{

struct VertexBufferUploadConfig
{
    std::string              vertexBufferName;
    std::vector<std::string> vertexAttributeNames;
    bool                    includePosition = false;
    // For a render-vertex buffer, gather these attributes through positionIndices.
    // The source remains in the unique-position domain; no duplicate CPU attribute is created.
    bool                    expandUniqueVertexAttributes = false;
};

// CPU-side dependency tracking for materialized GPU representations. Each buffer has its own
// stamp, so consuming a change for one view cannot hide it from another view or another system.
// Stamps describe data queued for upload, NOT GPU completion (owned by ResourceRegistry).
class MeshBufferCache
{
public:
    enum class Kind { RenderVertices, UniqueVertices, Indices, FaceGroups };

    struct Source
    {
        const Mesh *mesh = nullptr; // MeshStore keeps registered assets at stable addresses.
        uint32_t count = 0;
        Mesh::Revision topology = 0;
        Mesh::Revision positions = 0;
        Mesh::Revision faceGroups = 0;
        std::vector<Mesh::Revision> attributes;

        bool operator==(const Source &) const = default;
    };

    struct Stamp
    {
        Kind kind = Kind::RenderVertices;
        bool includePosition = false;
        bool expandUniqueVertexAttributes = false;
        std::vector<std::string> attributeNames;
        std::vector<Source> sources;

        bool operator==(const Stamp &) const = default;
    };

    static Stamp vertices(const std::vector<const Mesh *> &meshes, const VertexBufferUploadConfig &config,
                          bool uniqueVertices = false)
    {
        Stamp stamp{.kind = uniqueVertices ? Kind::UniqueVertices : Kind::RenderVertices,
                    .includePosition = config.includePosition,
                    .expandUniqueVertexAttributes = config.expandUniqueVertexAttributes,
                    .attributeNames = config.vertexAttributeNames};
        for (const Mesh *mesh : meshes)
        {
            Source source{.mesh = mesh,
                          .count = uniqueVertices ? mesh->uniquePositionCount() : mesh->vertexCount(),
                          .topology = uniqueVertices ? 0 : mesh->topologyRevision(),
                          .positions = config.includePosition ? mesh->positionsRevision() : 0};
            for (const auto &name : config.vertexAttributeNames)
            {
                source.attributes.push_back(uniqueVertices || config.expandUniqueVertexAttributes
                                                ? mesh->perUniqueVertexRevision(name)
                                                : mesh->perVertexRevision(name));
            }
            stamp.sources.push_back(std::move(source));
        }
        return stamp;
    }

    static Stamp indices(const std::vector<const Mesh *> &meshes)
    {
        Stamp stamp{.kind = Kind::Indices};
        for (const Mesh *mesh : meshes)
        {
            stamp.sources.push_back({.mesh = mesh, .count = mesh->faceCount(),
                                     .topology = mesh->topologyRevision()});
        }
        return stamp;
    }

    static Stamp faceGroups(const std::vector<const Mesh *> &meshes)
    {
        Stamp stamp{.kind = Kind::FaceGroups};
        for (const Mesh *mesh : meshes)
        {
            stamp.sources.push_back({.mesh = mesh, .count = mesh->faceCount(),
                                     .faceGroups = mesh->faceGroupsRevision()});
        }
        return stamp;
    }

    // Call only after initial upload/replacement has been successfully queued.
    void remember(const std::string &name, Stamp stamp) { m_stamps.insert_or_assign(name, std::move(stamp)); }

    template <typename QueueUpload>
    bool synchronize(const std::string &name, Stamp stamp, QueueUpload &&queueUpload)
    {
        const auto found = m_stamps.find(name);
        if (found == m_stamps.end()) throw std::logic_error("MeshBufferCache: unregistered buffer '" + name + "'");
        const Stamp &previous = found->second;
        if (previous == stamp) return false;

        // Updating contents must not silently invalidate allocations, layouts, or draw offsets.
        // Count/layout changes use the explicit wait/replace/rebind workflow instead.
        if (previous.kind != stamp.kind || previous.includePosition != stamp.includePosition ||
            previous.expandUniqueVertexAttributes != stamp.expandUniqueVertexAttributes ||
            previous.attributeNames != stamp.attributeNames || previous.sources.size() != stamp.sources.size())
        {
            throw std::logic_error("MeshBufferCache: buffer layout changed; replace '" + name + "'");
        }
        for (size_t i = 0; i < stamp.sources.size(); ++i)
        {
            if (previous.sources[i].count != stamp.sources[i].count)
            {
                throw std::logic_error("MeshBufferCache: mesh counts changed; rebuild geometry before syncing '" + name + "'");
            }
        }

        std::invoke(std::forward<QueueUpload>(queueUpload));
        remember(name, std::move(stamp));
        return true;
    }

private:
    std::unordered_map<std::string, Stamp> m_stamps;
};

} // namespace lr
