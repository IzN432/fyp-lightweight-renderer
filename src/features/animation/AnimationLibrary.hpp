#pragma once

#include "features/animation/AnimationClip.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lr
{

using AnimationClipHandle = uint32_t;

// Scene-owned, append-only animation assets. Handles remain stable for the lifetime of a scene and
// the global animation system addresses clips through these stable handles.
class AnimationLibrary
{
public:
    AnimationClipHandle add(AnimationClip clip)
    {
        const auto handle = static_cast<AnimationClipHandle>(m_clips.size());
        m_clips.push_back(std::move(clip));
        return handle;
    }

    AnimationClip &get(AnimationClipHandle handle)
    {
        if (handle >= m_clips.size()) throw std::out_of_range("Animation clip handle is out of range");
        return m_clips[handle];
    }
    const AnimationClip &get(AnimationClipHandle handle) const
    {
        if (handle >= m_clips.size()) throw std::out_of_range("Animation clip handle is out of range");
        return m_clips[handle];
    }

    const std::vector<AnimationClip> &clips() const { return m_clips; }
    size_t size() const { return m_clips.size(); }
    bool empty() const { return m_clips.empty(); }
    bool contains(AnimationClipHandle handle) const { return handle < m_clips.size(); }
    void clear() { m_clips.clear(); }

private:
    std::vector<AnimationClip> m_clips;
};

} // namespace lr
