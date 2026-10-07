#include "core/scene/MeshStore.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

MeshHandle MeshStore::add(Mesh mesh) { return add(std::move(mesh), generateUuid()); }

MeshHandle MeshStore::add(Mesh mesh, MeshId id)
{
    if (id.is_nil())
    {
        throw std::invalid_argument("A mesh cannot be stored under a nil ID");
    }
    if (m_handlesById.contains(id))
    {
        throw std::invalid_argument("A mesh with ID " + toString(id) + " already exists");
    }

    const MeshHandle handle = static_cast<MeshHandle>(m_meshes.size());
    m_meshes.push_back(std::make_unique<Mesh>(std::move(mesh)));
    m_ids.push_back(id);
    m_handlesById.emplace(id, handle);
    return handle;
}

Mesh &MeshStore::get(MeshHandle handle)
{
    if (handle >= m_meshes.size())
    {
        throw std::out_of_range("Mesh handle is out of range");
    }
    return *m_meshes[handle];
}

const Mesh &MeshStore::get(MeshHandle handle) const
{
    if (handle >= m_meshes.size())
    {
        throw std::out_of_range("Mesh handle is out of range");
    }
    return *m_meshes[handle];
}

MeshId MeshStore::idOf(MeshHandle handle) const
{
    if (handle >= m_ids.size())
    {
        throw std::out_of_range("Mesh handle is out of range");
    }
    return m_ids[handle];
}

std::optional<MeshHandle> MeshStore::find(MeshId id) const
{
    const auto found = m_handlesById.find(id);
    if (found == m_handlesById.end())
    {
        return std::nullopt;
    }
    return found->second;
}

} // namespace lr
