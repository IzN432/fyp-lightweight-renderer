#pragma once

#include "core/editor/command/Command.hpp"
#include "features/animation/AnimationLibrary.hpp"
#include "features/animation/AnimationTrack.hpp"

#include <cstddef>

namespace lr
{

class AnimationSystem;

// One edit of a track's contents from the Animation panel — adding or deleting a keyframe. A whole
// channel is carried either way rather than the one keyframe: a channel is a small value, and
// copying it keeps this command indifferent to how a track stores its keys.
//
// The target is named by clip handle and track index, both of which stay valid for as long as the
// clip and track are there; a command whose track has since gone does nothing.
class EditAnimationTrackCommand : public Command
{
public:
    EditAnimationTrackCommand(AnimationLibrary &library, AnimationClipHandle clip, std::size_t track,
                              AnimationChannel before, AnimationChannel after)
        : m_library(library), m_clip(clip), m_track(track), m_before(std::move(before)),
          m_after(std::move(after))
    {}

    void execute() override;
    void undo() override;

private:
    void restore(const AnimationChannel &channel);

    AnimationLibrary   &m_library;
    AnimationClipHandle m_clip;
    std::size_t         m_track;
    AnimationChannel    m_before;
    AnimationChannel    m_after;
};

// A track added from the Animation panel's "Add Animation Track?" prompt, carrying the keyframe the
// prompt seeds it with. Undo takes it off through the animation system, the same path the Delete
// shortcut uses, so playback bindings are released rather than left pointing at a track that has
// gone.
class AddAnimationTrackCommand : public Command
{
public:
    AddAnimationTrackCommand(AnimationSystem &system, AnimationLibrary &library, AnimationClipHandle clip,
                             AnimationChannel channel)
        : m_system(system), m_library(library), m_clip(clip), m_channel(std::move(channel))
    {}

    void execute() override;
    void undo() override;

private:
    AnimationSystem    &m_system;
    AnimationLibrary   &m_library;
    AnimationClipHandle m_clip;
    AnimationChannel    m_channel;
    // Where the track ended up, so undo removes the one this command added. Tracks are appended, so
    // this is the end of the list at the time it was added.
    std::size_t m_index = 0;
};

} // namespace lr
