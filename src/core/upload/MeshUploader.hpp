#pragma once

#include "core/framegraph/ResourceRegistry.hpp"
#include "core/loaders/Loader.hpp"
#include "core/scene/Mesh.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace lr
{

struct MeshUploadResult
{
    std::unordered_map<uint32_t, std::string>   vertexBufferNames;
    std::string                                 indexBufferName;
    std::string                                 faceGroupIndexBufferName;
    uint32_t                                    indexCount = 0;

    std::string                                 diffuseTextureArrayName;
    uint32_t                                    materialTextureCount = 0;

    std::string                                 specularTextureArrayName;
    std::string                                 normalTextureArrayName;
    std::string                                 roughnessTextureArrayName;
    std::string                                 metallicTextureArrayName;
    std::string                                 emissiveTextureArrayName;
};

class MeshUploader
{
public:
    explicit MeshUploader(ResourceRegistry &registry);

    MeshUploadResult upload(const Mesh &mesh,
                            const GpuMeshLayout &gpuLayout,
                            const std::vector<MaterialInfo> &materials,
                            const std::string &namePrefix = "mesh");

private:
    ResourceRegistry &m_registry;
};

} // namespace lr
