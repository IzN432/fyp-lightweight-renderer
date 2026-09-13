#pragma once

#include "core/scene/Mesh.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace lr
{

using MeshHandle = uint32_t;

// Owns mesh assets at stable addresses. Multiple MeshComponents may reference
// the same handle without duplicating geometry.
class MeshStore
{
public:
    MeshHandle add(Mesh mesh);

    Mesh       &get(MeshHandle handle);
    const Mesh &get(MeshHandle handle) const;

    uint32_t size() const { return static_cast<uint32_t>(m_meshes.size()); }

private:
    std::vector<std::unique_ptr<Mesh>> m_meshes;
};

} // namespace lr
