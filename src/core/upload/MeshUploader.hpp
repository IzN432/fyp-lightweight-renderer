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
struct MeshUploadResult
{
    std::unordered_map<uint32_t, std::string> vertexBufferNames;
    std::string indexBufferName;
    std::string faceGroupBufferName;
    uint32_t indexCount = 0;
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

    MeshUploadResult upload(const Mesh &mesh,
                            const GpuMeshLayout &gpuLayout,
                            const std::string &namePrefix = "mesh");

private:
    ResourceRegistry &m_registry;
};

} // namespace lr
