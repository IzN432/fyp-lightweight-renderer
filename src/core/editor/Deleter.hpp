#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <utility>

namespace lr
{

// Owns the editor's one current destructive target. Callers provide an opaque identity for
// presentation and a guarded action for deletion, so the Delete shortcut does not need to know
// every kind of resource that can be removed.
class Deleter
{
public:
    using Action = std::function<void()>;

    void select(std::string identity, Action action)
    {
        m_identity = std::move(identity);
        m_action   = std::move(action);
    }

    void clear()
    {
        m_identity.clear();
        m_action = {};
    }

    bool selected(std::string_view identity) const { return m_action && m_identity == identity; }
    bool hasSelection() const { return static_cast<bool>(m_action); }

    void erase()
    {
        // Clear before invoking: the action may select another target or otherwise re-enter this
        // controller, and destroying the currently executing std::function would be unsafe.
        Action action = std::move(m_action);
        m_identity.clear();
        if (action) action();
    }

private:
    std::string m_identity;
    Action      m_action;
};

} // namespace lr
