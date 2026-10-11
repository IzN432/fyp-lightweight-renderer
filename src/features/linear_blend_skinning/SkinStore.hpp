#pragma once

#include "features/linear_blend_skinning/Skin.hpp"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lr
{

// Owns shared skin bindings at stable addresses. Components retain dense session handles, while
// SkinId is serialized and resolved back to a handle when a scene is loaded.
class SkinStore
{
public:
    SkinHandle add(Scene &scene, std::vector<Joint> joints);
    SkinHandle add(Scene &scene, std::vector<Joint> joints, SkinId id);

    const SkinBinding &get(SkinHandle handle) const;
    SkinId idOf(SkinHandle handle) const;
    std::optional<SkinHandle> find(SkinId id) const;

    uint32_t size() const { return static_cast<uint32_t>(m_skins.size()); }
    void clear()
    {
        m_skins.clear();
        m_ids.clear();
        m_handlesById.clear();
    }

private:
    std::vector<std::unique_ptr<SkinBinding>> m_skins;
    std::vector<SkinId>                          m_ids;
    std::unordered_map<SkinId, SkinHandle>       m_handlesById;
};

} // namespace lr
