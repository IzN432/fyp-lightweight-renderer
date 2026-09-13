#include "core/scene/MeshStore.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

MeshHandle MeshStore::add(Mesh mesh)
{
    const MeshHandle handle = static_cast<MeshHandle>(m_meshes.size());
    m_meshes.push_back(std::make_unique<Mesh>(std::move(mesh)));
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

} // namespace lr
