#include "features/animation/AnimationTrackCommands.hpp"

#include "features/animation/AnimationSystem.hpp"

namespace lr
{

void EditAnimationTrackCommand::restore(const AnimationChannel &channel)
{
    if (!m_library.contains(m_clip))
    {
        return;
    }
    std::vector<AnimationChannel> &tracks = m_library.get(m_clip).tracks();
    if (m_track >= tracks.size())
    {
        return;
    }
    tracks[m_track] = channel;
}

void EditAnimationTrackCommand::execute() { restore(m_after); }

void EditAnimationTrackCommand::undo() { restore(m_before); }

void AddAnimationTrackCommand::execute()
{
    if (!m_library.contains(m_clip))
    {
        return;
    }
    // Stopped first for the same reason removeTrack does it: a clip whose channels change under a
    // running playback would leave the conflict table describing a track list that no longer exists.
    m_system.stop(m_clip);
    std::vector<AnimationChannel> &tracks = m_library.get(m_clip).tracks();
    tracks.push_back(m_channel);
    m_index = tracks.size() - 1;
}

void AddAnimationTrackCommand::undo()
{
    if (!m_library.contains(m_clip) || m_index >= m_library.get(m_clip).tracks().size())
    {
        return;
    }
    m_system.removeTrack(m_clip, m_index);
}

} // namespace lr
