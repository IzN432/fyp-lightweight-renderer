#pragma once

#include <functional>
#include <optional>
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
    using Guard = std::function<std::optional<std::string>()>;

    struct EraseResult
    {
        enum class Status
        {
            NoSelection,
            Blocked,
            Erased,
        };

        Status      status = Status::NoSelection;
        std::string message;
    };

    void select(std::string identity, Action action)
    {
        select(std::move(identity), {}, std::move(action));
    }

    // `guard` returns a reason when deletion is currently unavailable, or nullopt when the action
    // may proceed. A blocked target remains selected so retrying the shortcut remains meaningful.
    void select(std::string identity, Guard guard, Action action)
    {
        m_identity = std::move(identity);
        m_guard    = std::move(guard);
        m_action   = std::move(action);
    }

    void clear()
    {
        m_identity.clear();
        m_guard = {};
        m_action = {};
    }

    bool selected(std::string_view identity) const { return m_action && m_identity == identity; }
    bool hasSelection() const { return static_cast<bool>(m_action); }

    EraseResult erase()
    {
        if (!m_action)
        {
            return {};
        }
        if (m_guard)
        {
            if (std::optional<std::string> reason = m_guard())
            {
                return {.status = EraseResult::Status::Blocked, .message = std::move(*reason)};
            }
        }

        // Clear before invoking: the action may select another target or otherwise re-enter this
        // controller, and destroying the currently executing std::function would be unsafe.
        Action action = std::move(m_action);
        m_identity.clear();
        m_guard = {};
        action();
        return {.status = EraseResult::Status::Erased};
    }

private:
    std::string m_identity;
    Guard       m_guard;
    Action      m_action;
};

} // namespace lr
