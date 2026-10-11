#pragma once

#include <deque>
#include <optional>
#include <string>

namespace lr
{

// Queues failures raised by editor actions and presents them one at a time. Reporting is safe from
// input callbacks outside an ImGui frame; drawPopup() performs all ImGui work later in the GUI pass.
class ErrorReporter
{
public:
    void report(std::string title, std::string message);
    void drawPopup();

private:
    struct Error
    {
        std::string title;
        std::string message;
    };

    std::deque<Error>    m_pending;
    std::optional<Error> m_current;
};

} // namespace lr
