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

    // meshes and skins are parallel. A null Skin entry denotes an unskinned mesh and produces
    // zero influence counts for that mesh. This is intended to be called once after scene geometry
    // has been gathered, because the resulting dynamic palette buffer has a fixed capacity.
    SkinUploadResult upload(const std::vector<const Mesh *> &meshes, const std::vector<Skin *> &skins);

    // Re-packs the current matrices from the same palette ordering established by upload().
    void updateJointMatrices(const std::vector<Skin *> &skins);

    const std::string &influenceEntriesBufferName() const { return m_influenceEntriesBufferName; }
    const std::string &influenceOffsetsBufferName() const { return m_influenceOffsetsBufferName; }
    const std::string &positionIndicesBufferName() const { return m_positionIndicesBufferName; }
    const std::string &jointMatricesBufferName() const { return m_jointMatricesBufferName; }

private:
    ResourceRegistry &m_registry;

    std::string m_influenceEntriesBufferName;
    std::string m_influenceOffsetsBufferName;
    std::string m_positionIndicesBufferName;
    std::string m_jointMatricesBufferName;

    std::vector<uint32_t> m_expectedJointCounts;
    uint32_t              m_jointMatrixCapacity = 0;
    bool                  m_uploaded            = false;
};

} // namespace lr
