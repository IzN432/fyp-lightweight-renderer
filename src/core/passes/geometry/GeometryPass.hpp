#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/Transform.hpp"
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

        // One entry per mesh, parallel to vertexBufferUploadResult/indexBufferUploadResult's
        // singleMeshResults — read fresh every frame so dragging a Transform moves the mesh
        // immediately with no buffer/pass rebuild. A null entry means the mesh's vertex positions
        // are already baked into world space (e.g. AreaLightVisual's quads) and should be drawn
        // with an identity model matrix rather than double-transformed.
        std::vector<const Transform*> meshTransforms;
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
