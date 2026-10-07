#include "EditorShortcuts.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

void EditorShortcuts::add(KeyChord chord, Action action)
{
    if (!action)
    {
        throw std::invalid_argument("EditorShortcuts: shortcut action cannot be empty");
    }
    m_entries.push_back({chord, std::move(action)});
}

bool EditorShortcuts::dispatch(KeyChord pressed) const
{
    for (const Entry &entry : m_entries)
    {
        if (entry.chord == pressed)
        {
            entry.action();
            return true;
        }
    }
    return false;
}

} // namespace lr
