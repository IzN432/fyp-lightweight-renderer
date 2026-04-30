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
        
        std::string vertexAttributeBufferResourceName;
        std::string indexBufferResourceName;
        uint32_t    indexCount;

        std::string faceGroupBufferResourceName; // supposed to be optional but currently coded in

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
