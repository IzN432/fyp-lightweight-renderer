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

private:
    std::vector<std::unique_ptr<Command>> m_commandHistory;
    std::vector<std::unique_ptr<Command>> m_redoStack;
};

}