#include "CommandManager.hpp"

namespace lr
{

void CommandManager::executeCommand(std::unique_ptr<Command> command)
{
    command->execute();
    activeHistory().commands.push_back(std::move(command));
    activeHistory().redo.clear();
}
void CommandManager::appendCommandWithoutExecuting(std::unique_ptr<Command> command)
{
    activeHistory().commands.push_back(std::move(command));
    activeHistory().redo.clear();
}

void CommandManager::undo()
{
    History &history = activeHistory();
    if (!history.commands.empty())
    {
        std::unique_ptr<Command> command = std::move(history.commands.back());
        command->undo();
        history.commands.pop_back();
        history.redo.push_back(std::move(command));
    }
}

void CommandManager::redo()
{
    History &history = activeHistory();
    if (!history.redo.empty())
    {
        std::unique_ptr<Command> command = std::move(history.redo.back());
        history.redo.pop_back();
        command->execute();
        history.commands.push_back(std::move(command));
    }
}

void CommandManager::beginTemporaryHistory()
{
    m_histories.emplace_back();
}

void CommandManager::rollbackActiveHistory()
{
    History &history = activeHistory();
    while (!history.commands.empty())
    {
        history.commands.back()->undo();
        history.commands.pop_back();
    }
}

void CommandManager::cancelTemporaryHistory()
{
    if (!hasTemporaryHistory())
    {
        return;
    }
    rollbackActiveHistory();
    m_histories.pop_back();
}

void CommandManager::replaceTemporaryHistory(std::unique_ptr<Command> command)
{
    if (!hasTemporaryHistory())
    {
        executeCommand(std::move(command));
        return;
    }
    rollbackActiveHistory();
    m_histories.pop_back();
    executeCommand(std::move(command));
}

} // namespace lr
