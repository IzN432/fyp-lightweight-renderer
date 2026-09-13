#include "core/upload/SkinUploader.hpp"

#include "core/scene/Mesh.hpp"
#include "features/linear_blend_skinning/Skin.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace lr
{

SkinUploader::SkinUploader(ResourceRegistry &registry, std::string name)
    : m_registry(registry), m_influenceEntriesBufferName(name + "InfluenceEntries"),
      m_influenceOffsetsBufferName(name + "InfluenceOffsets"),
      m_positionIndicesBufferName(name + "PositionIndices"), m_jointMatricesBufferName(name + "JointMatrices")
{}

SkinUploadResult SkinUploader::upload(const std::vector<const Mesh *> &meshes,
                                      const std::vector<Skin *> &skins)
{
    if (m_uploaded)
    {
        throw std::logic_error("SkinUploader::upload may only be called once");
    }
    if (meshes.size() != skins.size())
    {
        throw std::invalid_argument("SkinUploader::upload: meshes and skins must be parallel");
    }

    SkinUploadResult result;
    result.drawInfos.resize(meshes.size());
    m_expectedJointCounts.resize(skins.size(), 0u);

    uint32_t totalPositionCount = 0;
    uint32_t totalVertexCount   = 0;
    uint32_t totalEntryCount    = 0;
    uint32_t totalJointCount    = 0;
    for (size_t i = 0; i < meshes.size(); ++i)
    {
        if (!meshes[i])
        {
            throw std::invalid_argument("SkinUploader::upload: mesh must not be null");
        }

        const Mesh &mesh = *meshes[i];
        totalPositionCount += mesh.uniquePositionCount();
        totalVertexCount += mesh.vertexCount();

        if (!skins[i])
        {
            continue;
        }
        if (!mesh.layout().vertexGroupsEnabled())
        {
            throw std::invalid_argument("SkinUploader::upload: a skinned mesh has no vertex groups");
        }

        const uint32_t jointCount = static_cast<uint32_t>(skins[i]->jointMatrices().size());
        result.drawInfos[i]       = {.paletteOffset = totalJointCount, .skinEnabled = true};
        m_expectedJointCounts[i]  = jointCount;
        totalJointCount += jointCount;
        totalEntryCount += static_cast<uint32_t>(mesh.rawGroupEntries().size());

        for (const VertexGroupEntry &entry : mesh.rawGroupEntries())
        {
            if (entry.groupIndex >= jointCount)
            {
                throw std::invalid_argument("SkinUploader::upload: vertex influence references a missing joint");
            }
        }
    }

    // Vulkan buffers cannot have size zero. Fallback elements also let every render pipeline use
    // the same descriptors when a scene contains no skinned mesh.
    std::vector<VertexGroupEntry> entries(std::max(totalEntryCount, 1u), VertexGroupEntry{0u, 0.0f});
    // Canonical CSR stores one trailing sentinel, allowing a position's influence range to be
    // read as offsets[position]..offsets[position + 1] without a separate counts buffer.
    std::vector<uint32_t> offsets(static_cast<size_t>(totalPositionCount) + 1, 0u);
    std::vector<uint32_t> positionIndices(std::max(totalVertexCount, 1u), 0u);

    uint32_t positionCursor = 0;
    uint32_t vertexCursor   = 0;
    uint32_t entryCursor    = 0;
    for (size_t i = 0; i < meshes.size(); ++i)
    {
        const Mesh &mesh = *meshes[i];

        for (uint32_t vertex = 0; vertex < mesh.vertexCount(); ++vertex)
        {
            positionIndices[vertexCursor + vertex] = positionCursor + mesh.positionIndices()[vertex];
        }

        if (skins[i])
        {
            const auto meshEntries = mesh.rawGroupEntries();
            const auto meshOffsets = mesh.rawGroupOffsets();
            std::memcpy(entries.data() + entryCursor, meshEntries.data(), meshEntries.size_bytes());

            for (uint32_t position = 0; position <= mesh.uniquePositionCount(); ++position)
            {
                offsets[positionCursor + position] = entryCursor + meshOffsets[position];
            }
            entryCursor += static_cast<uint32_t>(meshEntries.size());
        }
        else
        {
            for (uint32_t position = 0; position <= mesh.uniquePositionCount(); ++position)
            {
                offsets[positionCursor + position] = entryCursor;
            }
        }

        positionCursor += mesh.uniquePositionCount();
        vertexCursor += mesh.vertexCount();
    }

    m_registry.uploadBuffer(m_influenceEntriesBufferName, entries.data(), entries.size() * sizeof(entries[0]),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_registry.uploadBuffer(m_influenceOffsetsBufferName, offsets.data(), offsets.size() * sizeof(offsets[0]),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_registry.uploadBuffer(m_positionIndicesBufferName, positionIndices.data(),
                            positionIndices.size() * sizeof(positionIndices[0]), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    m_jointMatrixCapacity = std::max(totalJointCount, 1u);
    m_registry.registerDynamicBuffer(m_jointMatricesBufferName, m_jointMatrixCapacity * sizeof(glm::mat4),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    result.jointMatrixCount = totalJointCount;
    m_uploaded              = true;
    updateJointMatrices(skins);
    return result;
}

void SkinUploader::updateJointMatrices(const std::vector<Skin *> &skins)
{
    if (!m_uploaded)
    {
        throw std::logic_error("SkinUploader::updateJointMatrices called before upload");
    }
    if (skins.size() != m_expectedJointCounts.size())
    {
        throw std::invalid_argument("SkinUploader::updateJointMatrices: skin list size changed");
    }

    std::vector<glm::mat4> matrices;
    matrices.reserve(m_jointMatrixCapacity);
    for (size_t i = 0; i < skins.size(); ++i)
    {
        if (!skins[i])
        {
            if (m_expectedJointCounts[i] != 0)
            {
                throw std::invalid_argument("SkinUploader::updateJointMatrices: a skin was removed");
            }
            continue;
        }

        const auto &jointMatrices = skins[i]->jointMatrices();
        if (jointMatrices.size() != m_expectedJointCounts[i])
        {
            throw std::invalid_argument("SkinUploader::updateJointMatrices: joint count changed");
        }
        matrices.insert(matrices.end(), jointMatrices.begin(), jointMatrices.end());
    }

    if (matrices.empty())
    {
        matrices.emplace_back(1.0f);
    }
    m_registry.updateBuffer(m_jointMatricesBufferName, matrices.data(), matrices.size() * sizeof(matrices[0]));
}

} // namespace lr
