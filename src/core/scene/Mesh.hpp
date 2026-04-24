#pragma once

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace lr
{

// =============================================================================
// MeshLayout — definition phase
// =============================================================================

class MeshLayout
{
public:
    struct AttributeDesc
    {
        std::string     name    = "";   // unique identifier for this attribute
        size_t          stride  = 0;  // sizeof(T) for a T-typed attribute; auto-computed for group entries
        std::type_index type    = std::type_index(typeid(void));
    };

    // Per-vertex attributes (normals, UVs, tangents, custom, ...)
    template<typename T>
    MeshLayout &addPerVertexAttr(const std::string &name);

    // Per-face attributes
    template<typename T>
    MeshLayout &addPerFaceAttr(const std::string &name);

    template<typename T>
    MeshLayout &addFaceGroupAttr(const std::string &name);

    // Enable weighted vertex groups (CSR layout — each vertex holds a
    // variable-length list of (groupIndex, weight) pairs).
    // Use addVertexGroupAttr() to attach per-group data (e.g. bone transforms).
    MeshLayout &enableVertexGroups();

    template<typename T>
    MeshLayout &addVertexGroupAttr(const std::string &name);

    // Read-only accessors used by Mesh and GpuMeshLayout
    const std::vector<AttributeDesc> &perVertexAttrs()   const { return m_perVertex;   }
    const std::vector<AttributeDesc> &perFaceAttrs()     const { return m_perFace;     }
    const std::vector<AttributeDesc> &faceGroupAttrs()   const { return m_faceGroup;   }
    const std::vector<AttributeDesc> &vertexGroupAttrs() const { return m_vertexGroup; }

    bool vertexGroupsEnabled() const { return m_vertexGroupsEnabled; }

    const AttributeDesc *findPerVertexAttr(const std::string &name)   const;
    const AttributeDesc *findPerFaceAttr(const std::string &name)     const;
    const AttributeDesc *findFaceGroupAttr(const std::string &name)   const;
    const AttributeDesc *findVertexGroupAttr(const std::string &name) const;

private:
    template<typename T>
    static void pushAttr(std::vector<AttributeDesc> &vec, const std::string &name);

    std::vector<AttributeDesc> m_perVertex;
    std::vector<AttributeDesc> m_perFace;
    std::vector<AttributeDesc> m_faceGroup;
    std::vector<AttributeDesc> m_vertexGroup;

    bool m_vertexGroupsEnabled = false;
};

// =============================================================================
// GpuMeshLayout — maps named per-vertex attributes to Vulkan vertex input slots
// =============================================================================

class GpuMeshLayout
{
public:
    struct AttributeMapping
    {
        // must match a name registered in MeshLayout, empty when isPosition == true
        std::string name;
        // buffer index, same binding -> interleave
        uint32_t    binding;
        uint32_t    location;
        VkFormat    format;
        bool        isPosition = false;
    };

    explicit GpuMeshLayout(const MeshLayout &layout);

    // Map one per-vertex attribute to a Vulkan binding/location/format.
    GpuMeshLayout &map(std::string name, uint32_t binding, uint32_t location, VkFormat format);

    GpuMeshLayout &mapPosition(uint32_t binding, uint32_t location, VkFormat format);
    
    std::vector<VkVertexInputBindingDescription>   bindingDescriptions()   const;
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions() const;

    const std::vector<AttributeMapping> &mappings() const { return m_mappings; }

private:
    const MeshLayout             &m_layout;
    std::vector<AttributeMapping> m_mappings;
};

// =============================================================================
// Mesh — data container
// =============================================================================

struct VertexGroupEntry
{
    uint32_t groupIndex;
    float    weight;
};

class Mesh
{
public:
    explicit Mesh(const MeshLayout &layout);

    Mesh(const Mesh &)            = delete;
    Mesh &operator=(const Mesh &) = delete;
    Mesh(Mesh &&)                 = default;
    Mesh &operator=(Mesh &&)      = default;

    // -------------------------------------------------------------------------
    // First-class data
    // -------------------------------------------------------------------------

    // Positions — always required. Assigning this vector directly does NOT
    // update the internal vertex count; call setVertexCount() or use
    // setPerVertexArray() / setPerVertexAt() which keep counts in sync.
    std::vector<glm::vec3>  positions;
    std::vector<glm::uvec3> faces;       // triangle index triplets (uint32)

    // Per-face group membership — valid only when faceGroups are enabled.
    // Indexed by face index; each value is an index into the face group table.
    std::vector<uint32_t> faceGroups;

    // -------------------------------------------------------------------------
    // Count management
    // -------------------------------------------------------------------------

    // Explicit setters — lock the count and pre-allocate all registered
    // attribute stores for that domain.
    void setVertexCount(uint32_t count);
    void setFaceCount(uint32_t count);
    void setFaceGroupCount(uint32_t count);
    void setVertexGroupCount(uint32_t count);

    uint32_t vertexCount()      const { return m_vertexCount;      }
    uint32_t faceCount()        const { return m_faceCount;        }
    uint32_t faceGroupCount()   const { return m_faceGroupCount;   }
    uint32_t vertexGroupCount() const { return m_vertexGroupCount; }

    // -------------------------------------------------------------------------
    // Per-vertex attributes
    // -------------------------------------------------------------------------

    // Bulk set — implicitly sets vertex count from data.size() if not locked.
    template<typename T>
    void setPerVertexArray(const std::string &name, std::span<const T> data);

    // Single-index set — vertex count must already be set.
    template<typename T>
    void setPerVertexAt(const std::string &name, uint32_t index, const T &value);

    template<typename T>
    std::span<const T> getPerVertexArray(const std::string &name) const;

    template<typename T>
    T &perVertexAt(const std::string &name, uint32_t index);

    template<typename T>
    const T &perVertexAt(const std::string &name, uint32_t index) const;

    // -------------------------------------------------------------------------
    // Per-face attributes
    // -------------------------------------------------------------------------

    template<typename T>
    void setPerFaceArray(const std::string &name, std::span<const T> data);

    template<typename T>
    void setPerFaceAt(const std::string &name, uint32_t index, const T &value);

    template<typename T>
    std::span<const T> getPerFaceArray(const std::string &name) const;

    template<typename T>
    T &perFaceAt(const std::string &name, uint32_t index);

    template<typename T>
    const T &perFaceAt(const std::string &name, uint32_t index) const;

    // -------------------------------------------------------------------------
    // Per-face-group attributes  (requires enableFaceGroups() in layout)
    // -------------------------------------------------------------------------

    template<typename T>
    void setFaceGroupAttributeArray(const std::string &name, std::span<const T> data);

    template<typename T>
    void setFaceGroupAttributeAt(const std::string &name, uint32_t groupIndex, const T &value);

    template<typename T>
    std::span<const T> getFaceGroupAttributeArray(const std::string &name) const;

    template<typename T>
    T &getFaceGroupAttributeAt(const std::string &name, uint32_t groupIndex);

    template<typename T>
    const T &getFaceGroupAttributeAt(const std::string &name, uint32_t groupIndex) const;

    // -------------------------------------------------------------------------
    // Weighted vertex groups  (requires enableVertexGroups() in layout)
    // -------------------------------------------------------------------------

    // Set all (groupIndex, weight) pairs for one vertex.
    // Marks the CSR dirty; the flat arrays are rebuilt lazily on the first
    // rawGroupEntries/Offsets/Counts access after any call here.
    // Throws std::out_of_range if vertexIndex >= vertexCount().
    // Requires setVertexCount() to have been called explicitly beforehand.
    void setVertexGroups(uint32_t vertexIndex, std::span<const VertexGroupEntry> entries);

    std::span<const VertexGroupEntry> getVertexGroups(uint32_t vertexIndex) const;

    // Per-vertex-group attributes (bone name, rest pose, etc.)
    template<typename T>
    void setVertexGroupAttributeArray(const std::string &name, std::span<const T> data);

    template<typename T>
    void setVertexGroupAttributeAt(const std::string &name, uint32_t groupIndex, const T &value);

    template<typename T>
    std::span<const T> getVertexGroupAttributeArray(const std::string &name) const;

    template<typename T>
    T &getVertexGroupAttributeAt(const std::string &name, uint32_t groupIndex);

    template<typename T>
    const T &getVertexGroupAttributeAt(const std::string &name, uint32_t groupIndex) const;

    // -------------------------------------------------------------------------
    // Raw byte access — for GPU upload
    // -------------------------------------------------------------------------

    std::span<const std::byte> rawPerVertexData(const std::string &name)   const;
    std::span<const std::byte> rawPerFaceData(const std::string &name)     const;
    std::span<const std::byte> rawFaceGroupAttributeData(const std::string &name)   const;
    std::span<const std::byte> rawVertexGroupAttributeData(const std::string &name) const;

    // CSR buffers for weighted vertex group membership (skinning upload).
    // Triggers a CSR rebuild if entries have been modified since the last access.
    std::span<const VertexGroupEntry> rawGroupEntries() const;
    std::span<const uint32_t>         rawGroupOffsets() const;
    std::span<const uint32_t>         rawGroupCounts()  const;

    const MeshLayout &layout() const { return m_layout; }

private:
    // Internal attribute store — owns a flat byte buffer for one attribute
    struct AttributeStore
    {
        std::type_index        type     = std::type_index(typeid(void));
        size_t                 stride   = 0;
        std::vector<std::byte> data;
        uint32_t               count    = 0;
    };

    // Look up or create a store for a named attribute.
    // Throws if the name is not registered in the layout for that domain.
    AttributeStore &requireStore(std::unordered_map<std::string, AttributeStore> &stores,
                                 const std::vector<MeshLayout::AttributeDesc> &descs,
                                 const std::string &name);

    const AttributeStore &getStore(const std::unordered_map<std::string, AttributeStore> &stores,
                                   const std::string &name) const;

    void allocateDomain(std::unordered_map<std::string, AttributeStore> &stores,
                        const std::vector<MeshLayout::AttributeDesc> &descs,
                        uint32_t count);

    template<typename T>
    static void checkType(const AttributeStore &store);

    // Generic bulk-set used by all four domain variants
    template<typename T>
    void implSetArray(std::unordered_map<std::string, AttributeStore> &stores,
                      const std::vector<MeshLayout::AttributeDesc> &descs,
                      const std::string &name,
                      std::span<const T> data,
                      uint32_t &domainCount,
                      bool &explicitFlag);

    // Generic single-index set
    template<typename T>
    void implSetAt(std::unordered_map<std::string, AttributeStore> &stores,
                   const std::vector<MeshLayout::AttributeDesc> &descs,
                   const std::string &name,
                   uint32_t index,
                   const T &value);

    // Generic get-array
    template<typename T>
    std::span<const T> implGetArray(const std::unordered_map<std::string, AttributeStore> &stores,
                                    const std::string &name) const;

    // Generic mutable ref
    template<typename T>
    T &implGetAt(std::unordered_map<std::string, AttributeStore> &stores,
                 const std::vector<MeshLayout::AttributeDesc> &descs,
                 const std::string &name,
                 uint32_t index);

    const MeshLayout &m_layout;

    std::unordered_map<std::string, AttributeStore> m_perVertex;
    std::unordered_map<std::string, AttributeStore> m_perFace;
    std::unordered_map<std::string, AttributeStore> m_faceGroup;
    std::unordered_map<std::string, AttributeStore> m_vertexGroup;

    uint32_t m_vertexCount      = 0;
    uint32_t m_faceCount        = 0;
    uint32_t m_faceGroupCount   = 0;
    uint32_t m_vertexGroupCount = 0;

    bool m_vertexCountExplicit      = false;
    bool m_faceCountExplicit        = false;
    bool m_faceGroupCountExplicit   = false;
    bool m_vertexGroupCountExplicit = false;

    // Rebuilds m_groupEntries/Offsets/Counts from m_pendingGroupEntries.
    // Called lazily from rawGroup* accessors when m_csrDirty is set.
    void rebuildGroupCSR() const;

    // Build buffer: m_pendingGroupEntries[v] = entries for vertex v.
    // Sized to m_vertexCount on setVertexCount(); written by setVertexGroups().
    std::vector<std::vector<VertexGroupEntry>> m_pendingGroupEntries;

    // Flat CSR arrays — mutable because they are rebuilt lazily from const accessors.
    mutable std::vector<VertexGroupEntry> m_groupEntries;
    mutable std::vector<uint32_t>         m_groupOffsets;
    mutable std::vector<uint32_t>         m_groupCounts;
    mutable bool                          m_csrDirty = false;
};

// =============================================================================
// Template implementations
// =============================================================================

// --- MeshLayout ---

template<typename T>
void MeshLayout::pushAttr(std::vector<AttributeDesc> &vec, const std::string &name)
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "Mesh attributes must be trivially copyable");
    for (const auto &d : vec)
        if (d.name == name)
        {
            throw std::invalid_argument("MeshLayout: duplicate attribute name '" + name + "'");
        }
    vec.push_back({name, sizeof(T), std::type_index(typeid(T))});
}

template<typename T> MeshLayout &MeshLayout::addPerVertexAttr(const std::string &n)   { pushAttr<T>(m_perVertex,   n); return *this; }
template<typename T> MeshLayout &MeshLayout::addPerFaceAttr(const std::string &n)     { pushAttr<T>(m_perFace,     n); return *this; }
template<typename T> MeshLayout &MeshLayout::addFaceGroupAttr(const std::string &n)   { pushAttr<T>(m_faceGroup,   n); return *this; }
template<typename T> MeshLayout &MeshLayout::addVertexGroupAttr(const std::string &n) { pushAttr<T>(m_vertexGroup, n); return *this; }

// --- Mesh helpers ---

template<typename T>
void Mesh::checkType(const AttributeStore &store)
{
    if (store.type != std::type_index(typeid(T)))
    {
        throw std::invalid_argument("Mesh: attribute type mismatch for access");
    }
}

template<typename T>
void Mesh::implSetArray(std::unordered_map<std::string, AttributeStore> &stores,
                        const std::vector<MeshLayout::AttributeDesc> &descs,
                        const std::string &name,
                        std::span<const T> data,
                        uint32_t &domainCount,
                        bool &explicitFlag)
{
    auto &store = requireStore(stores, descs, name);
    checkType<T>(store);
    if (!explicitFlag)
        domainCount = static_cast<uint32_t>(data.size());
    else if (data.size() != domainCount)
        throw std::invalid_argument("Mesh: array size does not match locked domain count");
    store.count = domainCount;
    store.data.resize(domainCount * sizeof(T));
    std::memcpy(store.data.data(), data.data(), store.data.size());
}

template<typename T>
void Mesh::implSetAt(std::unordered_map<std::string, AttributeStore> &stores,
                     const std::vector<MeshLayout::AttributeDesc> &descs,
                     const std::string &name,
                     uint32_t index,
                     const T &value)
{
    auto &store = requireStore(stores, descs, name);
    checkType<T>(store);
    if (index >= store.count || store.data.empty())
    {
        throw std::out_of_range("Mesh: attribute index out of range");
    }
    std::memcpy(store.data.data() + index * sizeof(T), &value, sizeof(T));
}

template<typename T>
std::span<const T> Mesh::implGetArray(const std::unordered_map<std::string, AttributeStore> &stores,
                                       const std::string &name) const
{
    const auto &store = getStore(stores, name);
    checkType<T>(store);
    return {reinterpret_cast<const T *>(store.data.data()), store.count};
}

template<typename T>
T &Mesh::implGetAt(std::unordered_map<std::string, AttributeStore> &stores,
                   const std::vector<MeshLayout::AttributeDesc> &descs,
                   const std::string &name,
                   uint32_t index)
{
    auto &store = requireStore(stores, descs, name);
    checkType<T>(store);
    if (index >= store.count || store.data.empty())
    {
        throw std::out_of_range("Mesh: attribute index out of range");
    }
    return *reinterpret_cast<T *>(store.data.data() + index * sizeof(T));
}

// --- Per-vertex ---

template<typename T>
void Mesh::setPerVertexArray(const std::string &n, std::span<const T> d)
{ implSetArray(m_perVertex, m_layout.perVertexAttrs(), n, d, m_vertexCount, m_vertexCountExplicit); }

template<typename T>
void Mesh::setPerVertexAt(const std::string &n, uint32_t i, const T &v)
{ implSetAt(m_perVertex, m_layout.perVertexAttrs(), n, i, v); }

template<typename T>
std::span<const T> Mesh::getPerVertexArray(const std::string &n) const
{ return implGetArray<T>(m_perVertex, n); }

template<typename T>
T &Mesh::perVertexAt(const std::string &n, uint32_t i)
{ return implGetAt<T>(m_perVertex, m_layout.perVertexAttrs(), n, i); }

template<typename T>
const T &Mesh::perVertexAt(const std::string &n, uint32_t i) const
{ return implGetAt<T>(const_cast<std::unordered_map<std::string, AttributeStore> &>(m_perVertex),
                      m_layout.perVertexAttrs(), n, i); }

// --- Per-face ---

template<typename T>
void Mesh::setPerFaceArray(const std::string &n, std::span<const T> d)
{ implSetArray(m_perFace, m_layout.perFaceAttrs(), n, d, m_faceCount, m_faceCountExplicit); }

template<typename T>
void Mesh::setPerFaceAt(const std::string &n, uint32_t i, const T &v)
{ implSetAt(m_perFace, m_layout.perFaceAttrs(), n, i, v); }

template<typename T>
std::span<const T> Mesh::getPerFaceArray(const std::string &n) const
{ return implGetArray<T>(m_perFace, n); }

template<typename T>
T &Mesh::perFaceAt(const std::string &n, uint32_t i)
{ return implGetAt<T>(m_perFace, m_layout.perFaceAttrs(), n, i); }

template<typename T>
const T &Mesh::perFaceAt(const std::string &n, uint32_t i) const
{ return implGetAt<T>(const_cast<std::unordered_map<std::string, AttributeStore> &>(m_perFace),
                      m_layout.perFaceAttrs(), n, i); }

// --- Per-face-group ---

template<typename T>
void Mesh::setFaceGroupAttributeArray(const std::string &n, std::span<const T> d)
{ implSetArray(m_faceGroup, m_layout.faceGroupAttrs(), n, d, m_faceGroupCount, m_faceGroupCountExplicit); }

template<typename T>
void Mesh::setFaceGroupAttributeAt(const std::string &n, uint32_t i, const T &v)
{ implSetAt(m_faceGroup, m_layout.faceGroupAttrs(), n, i, v); }

template<typename T>
std::span<const T> Mesh::getFaceGroupAttributeArray(const std::string &n) const
{ return implGetArray<T>(m_faceGroup, n); }

template<typename T>
T &Mesh::getFaceGroupAttributeAt(const std::string &n, uint32_t i)
{ return implGetAt<T>(m_faceGroup, m_layout.faceGroupAttrs(), n, i); }

template<typename T>
const T &Mesh::getFaceGroupAttributeAt(const std::string &n, uint32_t i) const
{ return implGetAt<T>(const_cast<std::unordered_map<std::string, AttributeStore> &>(m_faceGroup),
                      m_layout.faceGroupAttrs(), n, i); }

// --- Per-vertex-group ---

template<typename T>
void Mesh::setVertexGroupAttributeArray(const std::string &n, std::span<const T> d)
{ implSetArray(m_vertexGroup, m_layout.vertexGroupAttrs(), n, d, m_vertexGroupCount, m_vertexGroupCountExplicit); }

template<typename T>
void Mesh::setVertexGroupAttributeAt(const std::string &n, uint32_t i, const T &v)
{ implSetAt(m_vertexGroup, m_layout.vertexGroupAttrs(), n, i, v); }

template<typename T>
std::span<const T> Mesh::getVertexGroupAttributeArray(const std::string &n) const
{ return implGetArray<T>(m_vertexGroup, n); }

template<typename T>
T &Mesh::getVertexGroupAttributeAt(const std::string &n, uint32_t i)
{ return implGetAt<T>(m_vertexGroup, m_layout.vertexGroupAttrs(), n, i); }

template<typename T>
const T &Mesh::getVertexGroupAttributeAt(const std::string &n, uint32_t i) const
{ return implGetAt<T>(const_cast<std::unordered_map<std::string, AttributeStore> &>(m_vertexGroup),
                      m_layout.vertexGroupAttrs(), n, i); }

}  // namespace lr
