#include "core/scene/Mesh.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace lr
{

// =============================================================================
// MeshLayout
// =============================================================================


MeshLayout &MeshLayout::enableVertexGroups()
{
    m_vertexGroupsEnabled = true;
    return *this;
}

static const MeshLayout::AttributeDesc *findInList(
    const std::vector<MeshLayout::AttributeDesc> &vec, const std::string &name)
{
    for (const auto &d : vec)
        if (d.name == name) return &d;
    return nullptr;
}

const MeshLayout::AttributeDesc *MeshLayout::findPerVertexAttr(const std::string &name)   const { return findInList(m_perVertex,   name); }
const MeshLayout::AttributeDesc *MeshLayout::findPerFaceAttr(const std::string &name)     const { return findInList(m_perFace,     name); }
const MeshLayout::AttributeDesc *MeshLayout::findFaceGroupAttr(const std::string &name)   const { return findInList(m_faceGroup,   name); }
const MeshLayout::AttributeDesc *MeshLayout::findVertexGroupAttr(const std::string &name) const { return findInList(m_vertexGroup, name); }

// =============================================================================
// GpuMeshLayout
// =============================================================================

GpuMeshLayout::GpuMeshLayout(const MeshLayout &layout) : m_layout(layout) {}

GpuMeshLayout &GpuMeshLayout::map(std::string name, uint32_t binding, uint32_t location, VkFormat format)
{
    if (!m_layout.findPerVertexAttr(name))
    {
        throw std::invalid_argument(
            "GpuMeshLayout: '" + name + "' is not a registered per-vertex attribute");
    }
    m_mappings.push_back({std::move(name), binding, location, format});
    return *this;
}

GpuMeshLayout &GpuMeshLayout::mapPosition(uint32_t binding, uint32_t location, VkFormat format)
{
    m_mappings.push_back({"", binding, location, format, true});
    return *this;
}

std::vector<VkVertexInputBindingDescription> GpuMeshLayout::bindingDescriptions() const
{
    // Sum strides of all attributes mapped to each binding.
    // For non-interleaved layouts (one attribute per binding) this equals sizeof(T).
    // For interleaved layouts (multiple attributes per binding) this is the total stride.
    std::unordered_map<uint32_t, uint32_t> bindingStrides;
    for (const auto &m : m_mappings)
    {
        uint32_t stride = m.isPosition
            ? static_cast<uint32_t>(sizeof(glm::vec3))
            : static_cast<uint32_t>(m_layout.findPerVertexAttr(m.name)->stride);
        bindingStrides[m.binding] += stride;
    }

    std::vector<VkVertexInputBindingDescription> result;
    result.reserve(bindingStrides.size());
    for (const auto &[binding, stride] : bindingStrides)
        result.push_back({binding, stride, VK_VERTEX_INPUT_RATE_VERTEX});
    return result;
}

std::vector<VkVertexInputAttributeDescription> GpuMeshLayout::attributeDescriptions() const
{
    // Offsets within each binding are accumulated in declaration order,
    // which matches the order mapPosition/map were called.
    std::unordered_map<uint32_t, uint32_t> bindingOffsets;

    std::vector<VkVertexInputAttributeDescription> result;
    result.reserve(m_mappings.size());
    for (const auto &m : m_mappings)
    {
        uint32_t stride = m.isPosition
            ? static_cast<uint32_t>(sizeof(glm::vec3))
            : static_cast<uint32_t>(m_layout.findPerVertexAttr(m.name)->stride);

        VkVertexInputAttributeDescription attr{};
        attr.location = m.location;
        attr.binding  = m.binding;
        attr.format   = m.format;
        attr.offset   = bindingOffsets[m.binding];
        result.push_back(attr);

        bindingOffsets[m.binding] += stride;
    }
    return result;
}

// =============================================================================
// Mesh — constructor
// =============================================================================

Mesh::Mesh(const MeshLayout &layout) : m_layout(layout) {}

// =============================================================================
// Mesh — private helpers
// =============================================================================

Mesh::AttributeStore &Mesh::requireStore(
    std::unordered_map<std::string, AttributeStore> &stores,
    const std::vector<MeshLayout::AttributeDesc> &descs,
    const std::string &name)
{
    auto it = stores.find(name);
    if (it != stores.end())
        return it->second;

    for (const auto &desc : descs)
    {
        if (desc.name == name)
        {
            auto [ins, _] = stores.emplace(name, AttributeStore{desc.type, desc.stride, {}, 0});
            return ins->second;
        }
    }

    throw std::invalid_argument("Mesh: '" + name + "' not registered in layout for this domain");
}

const Mesh::AttributeStore &Mesh::getStore(
    const std::unordered_map<std::string, AttributeStore> &stores,
    const std::string &name) const
{
    auto it = stores.find(name);
    if (it == stores.end())
    {
        throw std::invalid_argument("Mesh: '" + name + "' has no data — call a set* method first");
    }
    return it->second;
}

void Mesh::allocateDomain(
    std::unordered_map<std::string, AttributeStore> &stores,
    const std::vector<MeshLayout::AttributeDesc> &descs,
    uint32_t count)
{
    for (const auto &desc : descs)
    {
        auto &store  = stores[desc.name];
        store.type   = desc.type;
        store.stride = desc.stride;
        store.count  = count;
        store.data.assign(count * desc.stride, std::byte{0});
    }
}

void Mesh::rebuildGroupCSR() const
{
    const size_t n = m_pendingGroupEntries.size();
    m_groupOffsets.resize(n);
    m_groupCounts.resize(n);
    m_groupEntries.clear();

    for (size_t v = 0; v < n; ++v)
    {
        m_groupOffsets[v] = static_cast<uint32_t>(m_groupEntries.size());
        m_groupCounts[v]  = static_cast<uint32_t>(m_pendingGroupEntries[v].size());
        m_groupEntries.insert(m_groupEntries.end(),
                              m_pendingGroupEntries[v].begin(),
                              m_pendingGroupEntries[v].end());
    }

    m_csrDirty = false;
}

// =============================================================================
// Mesh — count management
// =============================================================================

void Mesh::setVertexCount(uint32_t count)
{
    m_vertexCount         = count;
    m_vertexCountExplicit = true;

    positions.resize(count);
    allocateDomain(m_perVertex, m_layout.perVertexAttrs(), count);

    if (m_layout.vertexGroupsEnabled())
    {
        m_pendingGroupEntries.resize(count);
        m_csrDirty = true;
    }
}

void Mesh::setFaceCount(uint32_t count)
{
    m_faceCount         = count;
    m_faceCountExplicit = true;

    faces.resize(count);
    allocateDomain(m_perFace, m_layout.perFaceAttrs(), count);

    faceGroups.resize(count, 0u);
}

void Mesh::setFaceGroupCount(uint32_t count)
{
    m_faceGroupCount         = count;
    m_faceGroupCountExplicit = true;

    allocateDomain(m_faceGroup, m_layout.faceGroupAttrs(), count);
}

void Mesh::setVertexGroupCount(uint32_t count)
{
    if (!m_layout.vertexGroupsEnabled())
    {
        throw std::logic_error("Mesh: vertex groups not enabled in layout");
    }

    m_vertexGroupCount         = count;
    m_vertexGroupCountExplicit = true;

    allocateDomain(m_vertexGroup, m_layout.vertexGroupAttrs(), count);
}

// =============================================================================
// Mesh — weighted vertex group membership (CSR)
// =============================================================================

void Mesh::setVertexGroups(uint32_t vertexIndex, std::span<const VertexGroupEntry> entries)
{
    if (!m_vertexCountExplicit)
    {
        throw std::logic_error(
            "Mesh: setVertexCount() must be called explicitly before setVertexGroups()");
    }
    if (vertexIndex >= m_vertexCount)
    {
        throw std::out_of_range(
            "Mesh: vertexIndex " + std::to_string(vertexIndex) +
            " >= vertexCount "  + std::to_string(m_vertexCount));
    }

    m_pendingGroupEntries[vertexIndex].assign(entries.begin(), entries.end());
    m_csrDirty = true;
}

std::span<const VertexGroupEntry> Mesh::getVertexGroups(uint32_t vertexIndex) const
{
    if (vertexIndex >= m_vertexCount)
    {
        throw std::out_of_range("Mesh: vertexIndex out of range in getVertexGroups()");
    }

    if (m_csrDirty)
        rebuildGroupCSR();

    return {m_groupEntries.data() + m_groupOffsets[vertexIndex], m_groupCounts[vertexIndex]};
}

// =============================================================================
// Mesh — raw byte access
// =============================================================================

std::span<const std::byte> Mesh::rawPerVertexData(const std::string &name) const
{
    return getStore(m_perVertex, name).data;
}

std::span<const std::byte> Mesh::rawPerFaceData(const std::string &name) const
{
    return getStore(m_perFace, name).data;
}

std::span<const std::byte> Mesh::rawFaceGroupAttributeData(const std::string &name) const
{
    return getStore(m_faceGroup, name).data;
}

std::span<const std::byte> Mesh::rawVertexGroupAttributeData(const std::string &name) const
{
    return getStore(m_vertexGroup, name).data;
}

std::span<const VertexGroupEntry> Mesh::rawGroupEntries() const
{
    if (m_csrDirty) rebuildGroupCSR();
    return m_groupEntries;
}

std::span<const uint32_t> Mesh::rawGroupOffsets() const
{
    if (m_csrDirty) rebuildGroupCSR();
    return m_groupOffsets;
}

std::span<const uint32_t> Mesh::rawGroupCounts() const
{
    if (m_csrDirty) rebuildGroupCSR();
    return m_groupCounts;
}

}  // namespace lr
