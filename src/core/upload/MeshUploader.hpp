#pragma once

#include "core/framegraph/ResourceRegistry.hpp"
#include "core/scene/Mesh.hpp"

#include <string>
#include <unordered_map>
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

struct VertexBufferUploadConfig
{
    std::string vertexBufferName;
    std::vector<std::string> vertexAttributeNames;
    bool includePosition = false; // whether to include the position attribute
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
 * Uploads a Mesh to GPU buffers according to a provided GpuMeshLayout, along with the materials used by the mesh.
 * The data is stored in the ResourceRegistry provided at initialization. The returned MeshUploadResult contains
 * the names of the registered resources for use in frame graph passes.
 */
class MeshUploader
{
public:
    explicit MeshUploader(ResourceRegistry &registry);

    VertexBufferUploadResult uploadVertexBuffer(const std::vector<const Mesh*> &meshes,
                                                const VertexBufferUploadConfig &config);
    
    IndexBufferUploadResult uploadIndexBuffer(const std::vector<const Mesh*> &meshes,
                           const IndexBufferUploadConfig &config);

    void uploadFaceGroupBuffer(const std::vector<const Mesh*> &meshes,
                               const FaceGroupBufferUploadConfig &config);

private:
    ResourceRegistry &m_registry;
};

} // namespace lr
