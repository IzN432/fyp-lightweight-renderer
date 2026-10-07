#pragma once

#include "core/scene/Mesh.hpp"
#include "core/utility/Uuid.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lr
{

using MeshHandle = uint32_t;

// The identity a mesh keeps across a save and a load, as opposed to MeshHandle, which only
// addresses this store's storage for the lifetime of the session.
using MeshId = Uuid;

// Owns mesh assets at stable addresses. Multiple MeshComponents may reference
// the same handle without duplicating geometry.
//
// Handles are per-session and dense, since they address this store's storage. Each mesh also
// carries a MeshId, which is the identity that survives a save and a load — persistence refers to
// meshes by MeshId and resolves back to a handle through find().
class MeshStore
{
public:
    MeshHandle add(Mesh mesh);
    // Adopts an identity a mesh already had, for loading a saved scene. Throws if that identity is
    // nil or already present in this store.
    MeshHandle add(Mesh mesh, MeshId id);

    Mesh       &get(MeshHandle handle);
    const Mesh &get(MeshHandle handle) const;

    MeshId                    idOf(MeshHandle handle) const;
    std::optional<MeshHandle> find(MeshId id) const;

    uint32_t size() const { return static_cast<uint32_t>(m_meshes.size()); }
    void     clear()
    {
        m_meshes.clear();
        m_ids.clear();
        m_handlesById.clear();
    }

private:
    std::vector<std::unique_ptr<Mesh>>     m_meshes;
    std::vector<MeshId>                    m_ids;
    std::unordered_map<MeshId, MeshHandle> m_handlesById;
};

} // namespace lr
