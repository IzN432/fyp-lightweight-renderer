#include "features/linear_blend_skinning/SkinStore.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

SkinHandle SkinStore::add(Scene &scene, std::vector<Joint> joints)
{
    return add(scene, std::move(joints), generateUuid());
}

SkinHandle SkinStore::add(Scene &scene, std::vector<Joint> joints, SkinId id)
{
    if (id.is_nil())
    {
        throw std::invalid_argument("A skin cannot be stored under a nil ID");
    }
    if (m_handlesById.contains(id))
    {
        throw std::invalid_argument("A skin with ID " + toString(id) + " already exists");
    }

    const SkinHandle handle = static_cast<SkinHandle>(m_skins.size());
    m_skins.push_back(std::make_unique<SkinBinding>(scene, std::move(joints), id));
    m_ids.push_back(id);
    m_handlesById.emplace(id, handle);
    return handle;
}

const SkinBinding &SkinStore::get(SkinHandle handle) const
{
    if (handle >= m_skins.size())
    {
        throw std::out_of_range("Skin handle is out of range");
    }
    return *m_skins[handle];
}

SkinId SkinStore::idOf(SkinHandle handle) const
{
    if (handle >= m_ids.size())
    {
        throw std::out_of_range("Skin handle is out of range");
    }
    return m_ids[handle];
}

std::optional<SkinHandle> SkinStore::find(SkinId id) const
{
    const auto found = m_handlesById.find(id);
    return found == m_handlesById.end() ? std::nullopt : std::optional<SkinHandle>(found->second);
}

} // namespace lr
