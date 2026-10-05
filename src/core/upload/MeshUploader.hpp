#pragma once

#include "core/framegraph/ResourceRegistry.hpp"
#include "core/scene/Mesh.hpp"
#include "core/upload/MeshBufferCache.hpp"

#include <string>
#include <vector>

namespace lr
{

/**
 * Result of uploading a Mesh, containing the name of the registered GPU resources
 * in the ResourceRegistry for use in frame graph passes. Also includes indexCount
 * for use in the draw call. For example usage, check out GeometryPass.cpp.
 */
struct VertexBufferUploadPerMeshResult
{
    uint32_t vertexOffset = 0;
};

struct VertexBufferUploadResult
{
    std::vector<VertexBufferUploadPerMeshResult> singleMeshResults; // details to separate the meshes up
};

struct IndexBufferUploadPerMeshResult
{
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
};

struct IndexBufferUploadResult
{
    std::vector<IndexBufferUploadPerMeshResult> singleMeshResults; // details to separate the meshes up
};

struct IndexBufferUploadConfig
{
    std::string indexBufferName;
};

struct FaceGroupBufferUploadConfig
{
    std::string faceGroupBufferName;
};

/**
 * Materializes named GPU views of canonical mesh data. Each view tracks its own source
 * revisions; ResourceRegistry owns allocations and transfer completion. Upload/replacement
 * returns draw ranges, while synchronization preserves allocations and ranges.
 */
class MeshUploader
{
public:
    explicit MeshUploader(ResourceRegistry &registry);

    VertexBufferUploadResult uploadVertexBuffer(const std::vector<const Mesh *> &meshes,
                                                const VertexBufferUploadConfig  &config);

    // Poll source revisions. Pack/queue only when this representation is stale; repeated edits
    // before this call collapse into one upload. Count/layout changes require replacement.
    bool synchronizeVertexBuffer(const std::vector<const Mesh *> &meshes, const VertexBufferUploadConfig &config);

    // Replaces an existing vertex buffer when the new mesh can have a different vertex count.
    VertexBufferUploadResult replaceVertexBuffer(const std::vector<const Mesh *> &meshes,
                                                 const VertexBufferUploadConfig  &config);

    // Packs each mesh's unique/deduped positions (mesh.positions() verbatim, not expanded through
    // positionIndices) together with named per-unique-vertex attributes (see
    // MeshLayout::addPerUniqueVertexAttr) into one interleaved buffer — the deduped-position-space
    // analogue of uploadVertexBuffer(). For consumers that index in deduped-position space rather
    // than per-UV-seam-corner space: VertexManager, SelectionManager, points-picking overlays.
    VertexBufferUploadResult uploadUniqueVertexBuffer(const std::vector<const Mesh *> &meshes,
                                                      const VertexBufferUploadConfig  &config);

    bool synchronizeUniqueVertexBuffer(const std::vector<const Mesh *> &meshes, const VertexBufferUploadConfig &config);

    // Like uploadUniqueVertexBuffer(), but for a buffer that already exists (see
    // ResourceRegistry::replaceUploadedBuffer()) — the new mesh list's total vertex count may
    // differ from what's currently allocated (e.g. the Scene Hierarchy selection switched to a
    // mesh with a different vertex count), unlike synchronizeUniqueVertexBuffer().
    VertexBufferUploadResult replaceUniqueVertexBuffer(const std::vector<const Mesh *> &meshes,
                                                        const VertexBufferUploadConfig  &config);

    IndexBufferUploadResult uploadIndexBuffer(const std::vector<const Mesh *> &meshes,
                                              const IndexBufferUploadConfig   &config);
    IndexBufferUploadResult replaceIndexBuffer(const std::vector<const Mesh *> &meshes,
                                               const IndexBufferUploadConfig   &config);
    bool synchronizeIndexBuffer(const std::vector<const Mesh *> &meshes, const IndexBufferUploadConfig &config);

    void uploadFaceGroupBuffer(const std::vector<const Mesh *> &meshes, const FaceGroupBufferUploadConfig &config);
    void replaceFaceGroupBuffer(const std::vector<const Mesh *> &meshes,
                                const FaceGroupBufferUploadConfig &config);
    bool synchronizeFaceGroupBuffer(const std::vector<const Mesh *> &meshes, const FaceGroupBufferUploadConfig &config);

private:
    ResourceRegistry &m_registry;
    MeshBufferCache m_bufferCache;
};

} // namespace lr
