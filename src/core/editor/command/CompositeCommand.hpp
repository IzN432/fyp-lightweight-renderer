#pragma once

#include "Command.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace lr
{

// Several commands that make up one gesture, undone as a unit. Built by
// CommandManager::endTransaction rather than by hand — see CommandTransaction.
//
// Undone in reverse, so each sub-command is put back from the state the one after it left behind,
// which is the only order that holds when two of them touch the same thing.
class CompositeCommand final : public Command
{
public:
    explicit CompositeCommand(std::vector<std::unique_ptr<Command>> commands)
        : m_commands(std::move(commands))
    {}

    void execute() override
    {
        for (std::unique_ptr<Command> &command : m_commands)
        {
            command->execute();
        }
    }

    void undo() override
    {
        for (auto command = m_commands.rbegin(); command != m_commands.rend(); ++command)
        {
            (*command)->undo();
        }
    }

private:
    std::vector<std::unique_ptr<Command>> m_commands;
};

} // namespace lr
