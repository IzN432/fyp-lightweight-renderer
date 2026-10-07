#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lr
{

// Move-only RAII handle returned by lifecycle-aware callback registrations. Destroying either the
// connection or its callback source is safe; destroying the connection first detaches the callback.
class CallbackConnection
{
public:
    CallbackConnection() = default;
    explicit CallbackConnection(std::function<void()> disconnect) : m_disconnect(std::move(disconnect)) {}
    ~CallbackConnection() { disconnect(); }

    CallbackConnection(const CallbackConnection &)            = delete;
    CallbackConnection &operator=(const CallbackConnection &) = delete;

    CallbackConnection(CallbackConnection &&other) noexcept
        : m_disconnect(std::exchange(other.m_disconnect, {}))
    {}

    CallbackConnection &operator=(CallbackConnection &&other) noexcept
    {
        if (this != &other)
        {
            disconnect();
            m_disconnect = std::exchange(other.m_disconnect, {});
        }
        return *this;
    }

    void disconnect()
    {
        if (!m_disconnect)
        {
            return;
        }
        auto disconnect = std::exchange(m_disconnect, {});
        disconnect();
    }

private:
    std::function<void()> m_disconnect;
};

// Callback storage for scoped registrations. Invocation snapshots the slots, so a callback may
// safely disconnect itself or another callback.
template <typename... Args> class CallbackList
{
public:
    using Callback = std::function<void(Args...)>;

    CallbackConnection connect(Callback callback)
    {
        const std::shared_ptr<Slot> slot = addSlot(std::move(callback));
        const std::weak_ptr<State>  state = m_state;
        const std::weak_ptr<Slot>   weakSlot = slot;
        return CallbackConnection([state, weakSlot] {
            const auto lockedState = state.lock();
            const auto lockedSlot  = weakSlot.lock();
            if (!lockedState || !lockedSlot)
            {
                return;
            }
            lockedSlot->connected = false;
            std::erase(lockedState->slots, lockedSlot);
        });
    }

    void invoke(Args... args)
    {
        const auto slots = m_state->slots;
        for (const auto &slot : slots)
        {
            if (slot->connected)
            {
                slot->callback(args...);
            }
        }
    }

private:
    struct Slot
    {
        Callback callback;
        bool     connected = true;
    };

    struct State
    {
        std::vector<std::shared_ptr<Slot>> slots;
    };

    std::shared_ptr<Slot> addSlot(Callback callback)
    {
        if (!callback)
        {
            throw std::invalid_argument("CallbackList: callback must not be empty");
        }
        auto slot = std::make_shared<Slot>(Slot{.callback = std::move(callback)});
        m_state->slots.push_back(slot);
        return slot;
    }

    std::shared_ptr<State> m_state = std::make_shared<State>();
};

} // namespace lr
