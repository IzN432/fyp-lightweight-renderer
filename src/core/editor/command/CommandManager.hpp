#pragma once

#include "Command.hpp"

#include <memory>
#include <vector>

namespace lr
{

class CommandManager
{
public:
    void executeCommand(std::unique_ptr<Command> command);
    void appendCommandWithoutExecuting(std::unique_ptr<Command> command);

    void undo();
    void redo();

    void beginTemporaryHistory();
    void cancelTemporaryHistory();
    void replaceTemporaryHistory(std::unique_ptr<Command> command);
    bool hasTemporaryHistory() const { return m_histories.size() > 1; }

private:
    struct History
    {
        std::vector<std::unique_ptr<Command>> commands;
        std::vector<std::unique_ptr<Command>> redo;
    };

    History &activeHistory() { return m_histories.back(); }
    void rollbackActiveHistory();

    std::vector<History> m_histories{1};
};

} // namespace lr
