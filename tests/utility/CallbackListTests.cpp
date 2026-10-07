#include "core/utility/CallbackList.hpp"

#include <cassert>
#include <utility>

namespace
{

void scopedConnectionsDetach()
{
    lr::CallbackList<int> callbacks;
    int                   total = 0;
    {
        auto connection = callbacks.connect([&](int value) {
            total += value;
        });
        callbacks.invoke(2);
        assert(total == 2);
    }
    callbacks.invoke(3);
    assert(total == 2);
}

void callbacksCanDisconnectDuringInvocation()
{
    lr::CallbackList<>  callbacks;
    lr::CallbackConnection second;
    int                    firstCalls  = 0;
    int                    secondCalls = 0;

    auto first = callbacks.connect([&] {
        ++firstCalls;
        second.disconnect();
    });
    second = callbacks.connect([&] {
        ++secondCalls;
    });

    callbacks.invoke();
    assert(firstCalls == 1);
    assert(secondCalls == 0);
}

void sourceMayDieBeforeConnection()
{
    lr::CallbackConnection connection;
    {
        lr::CallbackList<> callbacks;
        connection = callbacks.connect([] {});
    }
    connection.disconnect();
}

} // namespace

int main()
{
    scopedConnectionsDetach();
    callbacksCanDisconnectDuringInvocation();
    sourceMayDieBeforeConnection();
    return 0;
}
