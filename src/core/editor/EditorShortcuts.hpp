#pragma once

#include <functional>
#include <vector>

namespace lr
{

// A key chord an editor shortcut responds to. Only Ctrl is part of the match: Shift and Alt are
// ignored, which keeps chords like Shift+A available to whatever else wants them.
struct KeyChord
{
    int  key;
    bool ctrl = false;

    friend bool operator==(const KeyChord &, const KeyChord &) = default;
};

// Keyboard shortcuts contributed by the editor and by its tools. Owning them in one place lets a
// feature bring its own keys along with its state instead of the host holding them, and leaves the
// host a single place to decide whether the keyboard is the editor's at all this frame.
//
// Deliberately knows nothing about GLFW or ImGui: event filtering (key-press vs release, whether a
// text field has focus) belongs to whoever feeds it, so this stays testable without a UI context.
class EditorShortcuts
{
public:
    using Action = std::function<void()>;

    // Registering the same chord twice is allowed but pointless: dispatch stops at the first match,
    // so the earlier registration wins. The editor's own shortcuts register before any tool's.
    void add(KeyChord chord, Action action);

    // Runs the first shortcut matching `pressed`. Returns whether one consumed it, so a caller can
    // fall through to other input handling when nothing did.
    bool dispatch(KeyChord pressed) const;

private:
    struct Entry
    {
        KeyChord chord;
        Action   action;
    };

    std::vector<Entry> m_entries;
};

} // namespace lr
