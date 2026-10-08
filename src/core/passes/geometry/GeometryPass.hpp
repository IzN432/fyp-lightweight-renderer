#pragma once

#include "core/framegraph/FrameGraph.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/SceneDrawList.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"
#include "core/upload/MeshUploader.hpp"
#include "core/upload/SkinUploader.hpp"

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

        // The per-mesh draw state, replaced wholesale by setSceneGeometry() whenever SceneGpu
        // re-packs geometry. Everything else in this Config is a static resource binding set once
        // at construction.
        SceneDrawList draws;

        std::string                    indexBufferResourceName;
        std::string                    faceGroupBufferResourceName;
        std::string                    diffuseTextureArrayResourceName;
        std::string                    normalTextureArrayResourceName;
        std::string                    metallicRoughnessTextureArrayResourceName;
        std::string                    emissiveTextureArrayResourceName;
        std::string                    materialBufferResourceName;
        std::string                    skinInfluenceEntriesBufferResourceName;
        std::string                    skinInfluenceOffsetsBufferResourceName;
        std::string                    skinPositionIndicesBufferResourceName;
        std::string                    skinJointMatricesBufferResourceName;

        uint32_t materialCount;
    };

    explicit GeometryPass(Config cfg);

    void build(FrameGraph &fg, const GpuMeshLayout &layout) const;

    void setSceneGeometry(SceneDrawList draws) { m_cfg.draws = std::move(draws); }

    void setSkinningEnabled(bool enabled) { m_skinningEnabled = enabled; }
    bool isSkinningEnabled() const { return m_skinningEnabled; }

private:
    Config m_cfg;
    bool   m_skinningEnabled = true;
};

} // namespace lr
