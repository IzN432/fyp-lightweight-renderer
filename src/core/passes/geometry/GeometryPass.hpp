#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/scene/Mesh.hpp"
#include "core/upload/MeshUploader.hpp"

#include <vulkan/vulkan.h>

namespace lr
{

class GeometryPass
{
public:
    struct Config
    {
        std::string cameraBufferResourceName;

        std::unordered_map<uint32_t, std::string> vertexBufferResourceNames;
        VertexBufferUploadResult vertexBufferUploadResult;
        IndexBufferUploadResult indexBufferUploadResult;
        std::string indexBufferResourceName;
        std::string faceGroupBufferResourceName;
        std::string diffuseTextureArrayResourceName;
        std::string normalTextureArrayResourceName;
        std::string metallicRoughnessTextureArrayResourceName;
        std::string emissiveTextureArrayResourceName;
        std::string materialBufferResourceName;

        uint32_t materialCount;
    };

    explicit GeometryPass(Config cfg);

    void build(FrameGraph &fg, const GpuMeshLayout &layout) const;

private:
    Config m_cfg;
};

}  // namespace lr
