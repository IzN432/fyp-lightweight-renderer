#pragma once

#include "core/framegraph/ResourceRegistry.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace lr
{

class Mesh;
class Skin;

struct SkinDrawInfo
{
    uint32_t paletteOffset = 0;
    bool     skinEnabled   = false;
};

struct SkinUploadResult
{
    std::vector<SkinDrawInfo> drawInfos;
    uint32_t                  jointMatrixCount = 0;
};

// Owns the GPU resources shared by vertex-skinning consumers. Mesh influence data and the
// render-vertex-to-unique-position mapping are static; evaluated joint palettes are kept in a
// persistently mapped dynamic buffer so they can be replaced each frame.
class SkinUploader
{
public:
    explicit SkinUploader(ResourceRegistry &registry, std::string name = "skin");

    // Meshes and skins are parallel. A null Skin entry denotes an unskinned mesh. Calling upload
    // again replaces the existing buffers, allowing runtime scene imports to change topology and
    // joint capacity after the caller has synchronized with the GPU.
    SkinUploadResult upload(const std::vector<const Mesh *> &meshes, const std::vector<Skin *> &skins);

    // Re-packs the current matrices from the same palette ordering established by upload().
    // Structural topology/influence/binding changes require upload() after GPU synchronization.
    void updateJointMatrices(const std::vector<Skin *> &skins);

    const std::string &influenceEntriesBufferName() const { return m_influenceEntriesBufferName; }
    const std::string &influenceOffsetsBufferName() const { return m_influenceOffsetsBufferName; }
    const std::string &positionIndicesBufferName() const { return m_positionIndicesBufferName; }
    const std::string &jointMatricesBufferName() const { return m_jointMatricesBufferName; }

private:
    struct StaticMeshSource
    {
        const Mesh *mesh = nullptr;
        const Skin *skin = nullptr;
        uint32_t uniquePositionCount = 0;
        uint32_t vertexCount = 0;
        uint64_t topologyRevision = 0;
        uint64_t groupsRevision = 0;
    };

    ResourceRegistry &m_registry;

    std::string m_influenceEntriesBufferName;
    std::string m_influenceOffsetsBufferName;
    std::string m_positionIndicesBufferName;
    std::string m_jointMatricesBufferName;

    std::vector<uint32_t> m_expectedJointCounts;
    std::vector<StaticMeshSource> m_staticMeshSources;
    uint32_t              m_jointMatrixCapacity = 0;
    bool                  m_uploaded            = false;
};

} // namespace lr
